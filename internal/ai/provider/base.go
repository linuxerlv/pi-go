package provider

import (
	"context"

	"github.com/linuxerlv/pi-go/internal/ai"
)

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
		stream.Push(ai.ErrorEvent{Reason: ai.StopError, Error: fail(model, err.Error(), false)})
		return
	}

	sdkStream := startStream(ctx, params)
	tr := newTranslator(model)

	for sdkStream.Next() {
		if ctx.Err() != nil {
			stream.Push(ai.ErrorEvent{Reason: ai.StopAborted, Error: fail(model, "Operation aborted", true)})
			return
		}
		tr.handle(sdkStream.Current(), stream)
	}
	if err := sdkStream.Err(); err != nil {
		reason := ai.StopError
		aborted := false
		if ctx.Err() != nil {
			reason = ai.StopAborted
			aborted = true
		}
		stream.Push(ai.ErrorEvent{Reason: reason, Error: fail(model, err.Error(), aborted)})
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
