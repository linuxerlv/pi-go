package harness

import "sync"

// eventBus manages HarnessEvent subscribers. It is an internal component of
// AgentHarness, extracted to isolate the subscription/emit concern.
type eventBus struct {
	mu       sync.Mutex
	handlers []EventHandler
}

// subscribe registers a handler and returns an unsubscribe func.
func (b *eventBus) subscribe(handler EventHandler) func() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.handlers = append(b.handlers, handler)
	idx := len(b.handlers) - 1
	return func() {
		b.mu.Lock()
		defer b.mu.Unlock()
		if idx < len(b.handlers) {
			b.handlers[idx] = nil
		}
	}
}

// snapshot returns a copy of the current handler list for fan-out emission.
func (b *eventBus) snapshot() []EventHandler {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := make([]EventHandler, len(b.handlers))
	copy(out, b.handlers)
	return out
}

// emit fans an event out to all subscribers.
func (b *eventBus) emit(he HarnessEvent) {
	for _, handler := range b.snapshot() {
		if handler != nil {
			_ = handler(he)
		}
	}
}
