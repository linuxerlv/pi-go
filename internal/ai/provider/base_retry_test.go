package provider

import (
	"context"
	"errors"
	"io"
	"testing"
	"time"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// fakeStreamSource is a scriptable streamSource for driving runStreamCommon.
// Each call to startStream advances to the next scripted source. A scripted
// source yields the given events via Next()/Current(), then ends with err.
type fakeStreamSource struct {
	events []any // values returned by Current() while Next() is true
	err    error // returned by Err() after events exhausted
	idx    int
}

func (f *fakeStreamSource) Next() bool {
	if f.idx < len(f.events) {
		return true
	}
	return false
}
func (f *fakeStreamSource) Current() any {
	ev := f.events[f.idx]
	f.idx++
	return ev
}
func (f *fakeStreamSource) Err() error { return f.err }

// fakeTranslator records handle/finalize calls and controls terminal/snapshot.
type fakeTranslator struct {
	handled        []any
	finalized      bool
	terminalFlag   bool
	snapshotMsg    ai.AssistantMessage
	handleEvents   []ai.AssistantMessageEvent // events pushed to stream on handle
	finalizeEvents []ai.AssistantMessageEvent // events pushed on finalize
}

func (t *fakeTranslator) handle(event any, stream ai.AssistantMessageEventStream) {
	t.handled = append(t.handled, event)
	for _, e := range t.handleEvents {
		stream.Push(e)
	}
}
func (t *fakeTranslator) finalize(stream ai.AssistantMessageEventStream) {
	t.finalized = true
	for _, e := range t.finalizeEvents {
		stream.Push(e)
	}
}
func (t *fakeTranslator) snapshot() ai.AssistantMessage { return t.snapshotMsg }
func (t *fakeTranslator) terminalEmitted() bool         { return t.terminalFlag }

// runHarness wires up runStreamCommon with fakes and returns the stream plus
// the per-attempt source list and translator factory for inspection.
//
// handleEvents/finalizeEvents/terminalFlag are set on the translator BEFORE the
// producer goroutine starts, so handle() sees them regardless of scheduling
// (the producer can run before the caller resumes — a race that previously let
// handle fire with an empty handleEvents slice).
func runRetryTest(t *testing.T, sources []*fakeStreamSource, buildParamsErr error, handleEvents, finalizeEvents []ai.AssistantMessageEvent, terminalFlag bool) (ai.AssistantMessageEventStream, *fakeTranslator) {
	t.Helper()
	stream := ai.NewAssistantMessageEventStream()
	tr := &fakeTranslator{
		snapshotMsg:    ai.AssistantMessage{Provider: "mock", Model: "m", StopReason: ai.StopStop, Timestamp: ai.Now()},
		handleEvents:   handleEvents,
		finalizeEvents: finalizeEvents,
		terminalFlag:   terminalFlag,
	}
	srcIdx := 0
	startStream := func(ctx context.Context, params any) streamSource {
		if srcIdx >= len(sources) {
			t.Fatalf("startStream called more than scripted times (%d)", srcIdx+1)
		}
		s := sources[srcIdx]
		srcIdx++
		return s
	}
	buildParams := func(model ai.Model, c ai.Context, o *ai.SimpleStreamOptions) (any, error) {
		return nil, buildParamsErr
	}
	newTranslator := func(model ai.Model) translator { return tr }
	fail := func(model ai.Model, msg string, aborted bool) ai.AssistantMessage {
		return makeFailureMessage(model, ai.APIAnthropicMessages, "mock", msg, aborted)
	}
	go func() {
		defer stream.End(nil)
		runStreamCommon(context.Background(), ai.Model{ID: "m"}, ai.Context{}, nil, stream, buildParams, newTranslator, startStream, fail)
	}()
	return stream, tr
}

// drain consumes the stream until it is done, returning the terminal message
// and all events seen.
func drain(stream ai.AssistantMessageEventStream) (ai.AssistantMessage, []ai.AssistantMessageEvent) {
	var events []ai.AssistantMessageEvent
	var final ai.AssistantMessage
	for ev := range stream.Range {
		events = append(events, ev)
		if m, ok := ai.TerminalMessage(ev); ok {
			final = m
		}
	}
	return final, events
}

// TestRunStreamCommonRetriesUntilSuccess: first attempt fails with a retryable
// error before any event; second attempt succeeds. Verifies the loop retries
// and ultimately emits a DoneEvent (defect 6).
//
// Note: we use io.EOF as the retryable error rather than &anthropic.Error{} —
// runStreamCommon calls streamErr.Error() to build the failure message, and
// apierror.Error.Error() dereferences a nil Request (the SDK always populates
// it in production, but our bare struct literal would panic). io.EOF is
// retryable (isRetryable detects EOF) and has a safe Error(). The status-code
// retryability classification itself is covered by TestIsRetryableStatusCodes.
func TestRunStreamCommonRetriesUntilSuccess(t *testing.T) {
	sources := []*fakeStreamSource{
		{events: nil, err: io.EOF},                                 // attempt 0: fail before any event
		{events: []any{"chunk"}, err: nil},                        // attempt 1: success
	}
	stream, tr := runRetryTest(t, sources, nil,
		[]ai.AssistantMessageEvent{ai.TextDeltaEvent{Delta: "hi"}}, // handle: emit a delta
		[]ai.AssistantMessageEvent{ai.DoneEvent{Reason: ai.StopStop, Message: ai.AssistantMessage{Provider: "mock", Model: "m", StopReason: ai.StopStop, Timestamp: ai.Now()}}}, // finalize: emit done
		true, // terminalEmitted
	)
	// finalize emits the terminal done event so terminalEmitted path is covered.
	_ = tr

	final, events := drain(stream)
	if final.StopReason == ai.StopError {
		t.Fatalf("expected success after retry, got error: %s", final.ErrorMessage)
	}
	if !tr.finalized {
		t.Fatal("expected translator.finalize to be called on success")
	}
	// Must have seen at least one event from the second attempt.
	if len(events) == 0 {
		t.Fatal("expected events from successful attempt")
	}
}

// TestRunStreamCommonRetriesExhausted: every attempt fails with a retryable
// error before any event; after maxRetries the loop emits StopError.
func TestRunStreamCommonRetriesExhausted(t *testing.T) {
	// streamMaxRetries=2 => up to 3 attempts total (0,1,2).
	sources := []*fakeStreamSource{
		{err: io.EOF},
		{err: io.EOF},
		{err: io.EOF},
	}
	stream, _ := runRetryTest(t, sources, nil, nil, nil, false)
	final, _ := drain(stream)
	if final.StopReason != ai.StopError {
		t.Fatalf("expected StopError after retries exhausted, got %s", final.StopReason)
	}
	if final.ErrorMessage == "" {
		t.Fatal("expected non-empty error message")
	}
}

// TestRunStreamCommonNoRetryAfterHandled: once an event has been handled,
// a subsequent stream error must NOT retry (would duplicate output). The loop
// emits StopError immediately.
func TestRunStreamCommonNoRetryAfterHandled(t *testing.T) {
	sources := []*fakeStreamSource{
		{events: []any{"chunk"}, err: io.EOF}, // handled one event, then errored
	}
	stream, tr := runRetryTest(t, sources, nil,
		[]ai.AssistantMessageEvent{ai.TextDeltaEvent{Delta: "partial"}}, // handle: emit one delta
		nil,   // no finalize on error path
		false, // not terminal
	)
	final, events := drain(stream)
	if final.StopReason != ai.StopError {
		t.Fatalf("expected StopError (no retry after handled), got %s", final.StopReason)
	}
	// Exactly one event from the single handled chunk; no duplicate from a retry.
	var deltaCount int
	for _, e := range events {
		if _, ok := e.(ai.TextDeltaEvent); ok {
			deltaCount++
		}
	}
	if deltaCount != 1 {
		t.Fatalf("expected exactly 1 delta event (no retry), got %d", deltaCount)
	}
	if tr.finalized {
		t.Fatal("finalize must not run on error path")
	}
}

// TestRunStreamCommonNonRetryableErrorNoRetry: a non-retryable error is
// surfaced immediately without retry. errors.New is non-retryable and has a
// safe Error() (unlike a bare &anthropic.Error{} whose Error() needs Request).
func TestRunStreamCommonNonRetryableErrorNoRetry(t *testing.T) {
	nonRetryable := errors.New("bad request")
	sources := []*fakeStreamSource{
		{err: nonRetryable},
	}
	stream, _ := runRetryTest(t, sources, nil, nil, nil, false)
	final, _ := drain(stream)
	if final.StopReason != ai.StopError {
		t.Fatalf("expected StopError, got %s", final.StopReason)
	}
}

// TestRunStreamCommonBuildParamsErrorNoRetry: a buildParams error is surfaced
// immediately (not transient), with no stream attempt.
func TestRunStreamCommonBuildParamsErrorNoRetry(t *testing.T) {
	// No sources scripted; startStream must never be called.
	stream, _ := runRetryTest(t, nil, errors.New("bad params"), nil, nil, false)
	final, _ := drain(stream)
	if final.StopReason != ai.StopError {
		t.Fatalf("expected StopError, got %s", final.StopReason)
	}
	if final.ErrorMessage != "bad params" {
		t.Fatalf("expected 'bad params' message, got %q", final.ErrorMessage)
	}
}

// TestRunStreamCommonCtxCancelAborts: if the context is cancelled, the loop
// emits StopAborted and does not retry.
func TestRunStreamCommonCtxCancelAborts(t *testing.T) {
	stream := ai.NewAssistantMessageEventStream()
	tr := &fakeTranslator{
		snapshotMsg: ai.AssistantMessage{Provider: "mock", Model: "m", StopReason: ai.StopStop, Timestamp: ai.Now()},
	}
	ctx, cancel := context.WithCancel(context.Background())
	// A source whose Next() blocks until ctx is cancelled would be ideal; instead
	// we cancel before the run and feed a retryable error so the retry path's
	// ctx.Done() select fires.
	cancel()
	sources := []*fakeStreamSource{{err: io.EOF}}
	srcIdx := 0
	startStream := func(_ context.Context, _ any) streamSource {
		s := sources[srcIdx]
		srcIdx++
		return s
	}
	go func() {
		defer stream.End(nil)
		runStreamCommon(ctx, ai.Model{ID: "m"}, ai.Context{}, nil, stream,
			func(ai.Model, ai.Context, *ai.SimpleStreamOptions) (any, error) { return nil, nil },
			func(ai.Model) translator { return tr },
			startStream,
			func(model ai.Model, msg string, aborted bool) ai.AssistantMessage {
				return makeFailureMessage(model, ai.APIAnthropicMessages, "mock", msg, aborted)
			})
	}()
	final, _ := drain(stream)
	if final.StopReason != ai.StopAborted {
		t.Fatalf("expected StopAborted on ctx cancel, got %s", final.StopReason)
	}
}

// TestRetryBackoffValues verifies the backoff schedule (defect 6).
func TestRetryBackoffValues(t *testing.T) {
	if got := retryBackoff(0); got != 500*time.Millisecond {
		t.Fatalf("retryBackoff(0) = %v, want 500ms", got)
	}
	if got := retryBackoff(1); got != 1*time.Second {
		t.Fatalf("retryBackoff(1) = %v, want 1s", got)
	}
}
