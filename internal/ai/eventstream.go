package ai

import "sync"

// EventStream is a generic async-iterable event stream with a final result. It
// mirrors @earendil-works/pi-ai's EventStream<T,R>: a queue of events plus a
// future resolved when a terminal event (determined by isComplete) is pushed or
// when End is called with a result.
//
// Producers Push events (and at most one terminal event), then optionally call
// End. Consumers iterate via Range and await the final value via Result.
type EventStream[T any, R any] struct {
	isComplete    func(T) bool
	extractResult func(T) R

	mu       sync.Mutex
	queue    []T
	waiters  []chan T
	done     bool
	resultCh chan R
	once     sync.Once
}

// NewEventStream constructs an EventStream. isComplete identifies the terminal
// event; extractResult derives the final value from it.
func NewEventStream[T any, R any](isComplete func(T) bool, extractResult func(T) R) *EventStream[T, R] {
	return &EventStream[T, R]{
		isComplete:    isComplete,
		extractResult: extractResult,
		resultCh:      make(chan R, 1),
	}
}

// Push delivers an event to consumers. If the event satisfies isComplete, the
// final result is resolved and the stream is marked done. Push on a done
// stream is a no-op.
func (s *EventStream[T, R]) Push(event T) {
	s.mu.Lock()
	if s.done {
		// Terminal already seen; resolve result if this is terminal and not yet set.
		if s.isComplete(event) {
			s.resolveOnce(s.extractResult(event))
		}
		s.mu.Unlock()
		return
	}
	if s.isComplete(event) {
		s.done = true
		s.resolveOnce(s.extractResult(event))
	}

	var waiter chan T
	if len(s.waiters) > 0 {
		waiter = s.waiters[0]
		s.waiters = s.waiters[1:]
	}
	if waiter != nil {
		// Hand the event directly to the waiting consumer without buffering.
		s.mu.Unlock()
		waiter <- event
		return
	}
	s.queue = append(s.queue, event)
	s.mu.Unlock()
}

// End marks the stream as done and resolves the result if result is non-nil.
// Waiters blocked in Range are released. If result is nil and no terminal
// event has resolved the result, the result channel is closed without a value
// (Result returns ok=false).
func (s *EventStream[T, R]) End(result *R) {
	s.mu.Lock()
	s.done = true
	if result != nil {
		r := *result
		s.mu.Unlock()
		s.resolveOnce(r)
	} else {
		s.mu.Unlock()
		s.resolveZero()
	}
	waiters := s.waiters
	s.waiters = nil
	for _, w := range waiters {
		close(w)
	}
}

// resolveOnce sends r on the result channel exactly once (whether via a
// terminal event or End with a value), then closes it.
func (s *EventStream[T, R]) resolveOnce(r R) {
	s.once.Do(func() {
		s.resultCh <- r
		close(s.resultCh)
	})
}

// resolveZero closes the result channel without sending a value, signalling
// "stream ended without a terminal result". No-op if already resolved.
func (s *EventStream[T, R]) resolveZero() {
	s.once.Do(func() {
		close(s.resultCh)
	})
}

// Range yields events from the stream until it is done and drained. The ctx
// cancellation is the caller's responsibility via the surrounding stream call;
// Range itself blocks until events arrive or the stream ends.
//
// Iteration order is push order. Range returns when the stream is done and no
// buffered events remain.
func (s *EventStream[T, R]) Range(yield func(T) bool) {
	for {
		s.mu.Lock()
		if len(s.queue) > 0 {
			event := s.queue[0]
			s.queue = s.queue[1:]
			s.mu.Unlock()
			if !yield(event) {
				return
			}
			continue
		}
		if s.done {
			s.mu.Unlock()
			return
		}
		// No buffered events and not done: wait for the next push or end.
		w := make(chan T, 1)
		s.waiters = append(s.waiters, w)
		s.mu.Unlock()
		event, ok := <-w
		if !ok {
			// End closed the waiter; stream is done, drain anything left.
			continue
		}
		if !yield(event) {
			return
		}
	}
}

// Result returns a receive-only channel that yields the final value once. It is
// closed after the value is sent, so a closed channel without a value means the
// stream ended without a terminal event (e.g. End(nil)).
func (s *EventStream[T, R]) Result() <-chan R {
	return s.resultCh
}

// AssistantMessageEventStream is an EventStream of AssistantMessageEvent whose
// final value is the terminal AssistantMessage (from a done or error event).
type AssistantMessageEventStream = *EventStream[AssistantMessageEvent, AssistantMessage]

// NewAssistantMessageEventStream constructs an AssistantMessageEventStream.
func NewAssistantMessageEventStream() AssistantMessageEventStream {
	return NewEventStream[AssistantMessageEvent, AssistantMessage](
		func(e AssistantMessageEvent) bool { return IsTerminal(e) },
		func(e AssistantMessageEvent) AssistantMessage {
			m, _ := TerminalMessage(e)
			return m
		},
	)
}
