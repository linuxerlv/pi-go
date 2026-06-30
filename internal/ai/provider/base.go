package provider

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/linuxerlv/pi-go/internal/ai"
	openai "github.com/openai/openai-go/v3"
)

// streamRetryConfig tunes the conservative retry policy: only failures that
// occur before the stream has produced any event are retried, so a retry never
// produces duplicate/interleaved output. maxRetries is the number of additional
// attempts (so 2 = up to 3 total calls).
const (
	streamMaxRetries = 2
)

// retryBackoff returns the sleep before the given attempt (0-based): 500ms, 1s.
func retryBackoff(attempt int) time.Duration {
	return time.Duration(500<<attempt) * time.Millisecond
}

// translator is the strategy interface implemented by each provider's stream
// event translator. baseProvider.runStreamCommon drives the SDK stream loop
// and delegates per-event handling to a translator — the Template Method
// pattern, Go-style (composition + interface injection instead of inheritance).
type translator interface {
	// handle processes one SDK event, accumulating state and pushing pi events.
	handle(event any, stream ai.AssistantMessageEventStream)
	// finalize decodes any pending tool-call args and emits *_end events after
	// the stream ends (some gateways omit stop events).
	finalize(stream ai.AssistantMessageEventStream)
	// snapshot returns the current partial AssistantMessage.
	snapshot() ai.AssistantMessage
	// terminalEmitted reports whether handle already pushed a done/error event.
	terminalEmitted() bool
}

// failureFn builds an error AssistantMessage for a provider.
type failureFn func(model ai.Model, msg string, aborted bool) ai.AssistantMessage

// makeFailureMessage builds a failure AssistantMessage for a given provider. It
// removes the per-provider failureMessage duplication: every provider's failure
// message is identical except for the API/Provider tags.
func makeFailureMessage(model ai.Model, api ai.Api, provider, msg string, aborted bool) ai.AssistantMessage {
	reason := ai.StopError
	if aborted {
		reason = ai.StopAborted
	}
	return ai.AssistantMessage{
		Content:      []ai.ContentBlock{ai.TextContent{Type: "text", Text: ""}},
		API:          api,
		Provider:     provider,
		Model:        model.ID,
		StopReason:   reason,
		ErrorMessage: msg,
		Timestamp:    ai.Now(),
	}
}

// runStreamCommon is the Template Method: it owns the stream-loop skeleton
// (params-failure handling, Next iteration, ctx abort, sdkStream.Err, finalize,
// synthesize DoneEvent) shared by all providers. The variant steps — building
// provider params and constructing a translator — are injected as functions.
//
//   buildParams     : convert pi Context -> provider-specific request params
//   newTranslator   : create a fresh translator for this run
//   startStream     : kick off the provider's streaming call, returning a
//                     stream with Next()/Current()/Err() semantics
//
// The provider's stream type is captured generically via the streamSource
// interface so the skeleton stays provider-agnostic.
func runStreamCommon(
	ctx context.Context,
	model ai.Model,
	callCtx ai.Context,
	options *ai.SimpleStreamOptions,
	stream ai.AssistantMessageEventStream,
	buildParams func(model ai.Model, context ai.Context, options *ai.SimpleStreamOptions) (any, error),
	newTranslator func(model ai.Model) translator,
	startStream func(ctx context.Context, params any) streamSource,
	fail failureFn,
) {
	params, err := buildParams(model, callCtx, options)
	if err != nil {
		// Param-conversion errors are not transient; do not retry.
		stream.Push(ai.ErrorEvent{Reason: ai.StopError, Error: fail(model, err.Error(), false)})
		return
	}

	// Conservative retry loop: only retry when the stream failed before it
	// yielded any event (handled == false). Once a single event has been
	// handled, partial output may already have been emitted to the consumer, so
	// retrying could interleave/duplicate output — we surface the error
	// instead. This covers the common transient case (connection refused,
	// 429, 5xx, DNS) which typically fails at stream establishment.
	var sdkStream streamSource
	var tr translator
	for attempt := 0; ; attempt++ {
		sdkStream = startStream(ctx, params)
		tr = newTranslator(model)
		handled := false
		for sdkStream.Next() {
			if ctx.Err() != nil {
				stream.Push(ai.ErrorEvent{Reason: ai.StopAborted, Error: fail(model, "Operation aborted", true)})
				return
			}
			tr.handle(sdkStream.Current(), stream)
			handled = true
		}
		streamErr := sdkStream.Err()
		if streamErr == nil {
			break // clean end; proceed to finalize below.
		}
		// ctx cancelled takes precedence over retry.
		if ctx.Err() != nil {
			stream.Push(ai.ErrorEvent{Reason: ai.StopAborted, Error: fail(model, "Operation aborted", true)})
			return
		}
		if !handled && isRetryable(streamErr) && attempt < streamMaxRetries {
			select {
			case <-ctx.Done():
				stream.Push(ai.ErrorEvent{Reason: ai.StopAborted, Error: fail(model, "Operation aborted", true)})
				return
			case <-time.After(retryBackoff(attempt)):
			}
			continue // fresh stream + fresh translator
		}
		stream.Push(ai.ErrorEvent{Reason: ai.StopError, Error: fail(model, streamErr.Error(), false)})
		return
	}
	tr.finalize(stream)
	if !tr.terminalEmitted() {
		msg := tr.snapshot()
		if msg.StopReason == "" {
			msg.StopReason = ai.StopStop
		}
		stream.Push(ai.DoneEvent{Reason: msg.StopReason, Message: msg})
	}
}

// isRetryable reports whether a stream error is worth retrying. It recognizes
// transient network failures and HTTP 429 / 5xx from either SDK's API error
// type (both expose StatusCode as a field). Non-transient errors (4xx other
// than 429, auth errors, param errors) are not retried.
//
// This is intentionally cross-platform: it avoids syscall errno sentinels
// (ECONNRESET etc. are Unix-only and absent from syscall on Windows). Gateway
// connection drops typically surface as io.EOF / io.ErrUnexpectedEOF or a
// net.Error timeout, both of which are detected below.
func isRetryable(err error) bool {
	if err == nil {
		return false
	}
	// SDK API errors carry an HTTP status code.
	var anthErr *anthropic.Error
	if errors.As(err, &anthErr) {
		return retryableStatus(anthErr.StatusCode)
	}
	var oaiErr *openai.Error
	if errors.As(err, &oaiErr) {
		return retryableStatus(oaiErr.StatusCode)
	}
	// Transient network-layer errors: timeouts (covers dial/read/write timeouts
	// and TLS handshake deadlines).
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}
	// Gateways/proxies that close the connection mid-stream (before any event)
	// surface as unexpected EOF. Retry these.
	if errors.Is(err, io.ErrUnexpectedEOF) || errors.Is(err, io.EOF) {
		return true
	}
	// Temporary net operation errors (covers connection refused/reset on the
	// platforms that surface them via net.OpError.Temporary).
	var opErr *net.OpError
	if errors.As(err, &opErr) && opErr.Temporary() {
		return true
	}
	return false
}

// retryableStatus classifies an HTTP status as retryable: 429 (rate limit) and
// 5xx (server errors, including 502/503/504 from gateways).
func retryableStatus(statusCode int) bool {
	if statusCode == http.StatusTooManyRequests {
		return true
	}
	return statusCode >= 500 && statusCode < 600
}

// streamSource is the minimal streaming interface the skeleton depends on.
// Each provider wraps its SDK stream type (*ssestream.Stream[T]) in a small
// adapter satisfying this interface, so the skeleton stays generic over T.
type streamSource interface {
	Next() bool
	Current() any
	Err() error
}

// sseStreamAdapter wraps the SDK's *ssestream.Stream[T] so it satisfies
// streamSource (whose Current returns any). T is the provider's chunk/event
// union type.
type sseStreamAdapter[T any] struct {
	next    func() bool
	current func() T
	err     func() error
}

func (a *sseStreamAdapter[T]) Next() bool   { return a.next() }
func (a *sseStreamAdapter[T]) Current() any { return a.current() }
func (a *sseStreamAdapter[T]) Err() error   { return a.err() }

// newSSEAdapter builds an adapter from the three SDK stream methods.
func newSSEAdapter[T any](next func() bool, current func() T, err func() error) streamSource {
	return &sseStreamAdapter[T]{next: next, current: current, err: err}
}

// snapshotBase returns a copy of partial with content set from blocks. All
// three translators shared this exact body before extraction.
func snapshotBase(partial ai.AssistantMessage, blocks []ai.ContentBlock) ai.AssistantMessage {
	msg := partial
	msg.Content = append([]ai.ContentBlock(nil), blocks...)
	return msg
}
