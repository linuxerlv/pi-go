package ai

// AssistantMessageEvent is the streaming event protocol emitted by provider
// adapters. Mirrors @earendil-works/pi-ai's AssistantMessageEvent union.
//
// Providers emit `start` with the first partial AssistantMessage, then a series
// of *_delta/*_end events (each carrying the updated partial), and terminate
// with either `done` (success) or `error` (failure). Errors must NOT be thrown;
// they are encoded as an `error` event whose AssistantMessage has stopReason
// "error" or "aborted" and a populated ErrorMessage.
type AssistantMessageEvent interface {
	assistantMessageEvent()
	EventType() string
}

// StartEvent is emitted once, with the initial partial AssistantMessage.
type StartEvent struct {
	Partial AssistantMessage
}

// TextStartEvent marks the beginning of a text content block.
type TextStartEvent struct {
	ContentIndex int
	Partial      AssistantMessage
}

// TextDeltaEvent carries an incremental text chunk.
type TextDeltaEvent struct {
	ContentIndex int
	Delta        string
	Partial      AssistantMessage
}

// TextEndEvent marks the end of a text content block.
type TextEndEvent struct {
	ContentIndex int
	Content      string
	Partial      AssistantMessage
}

// ThinkingStartEvent marks the beginning of a thinking content block.
type ThinkingStartEvent struct {
	ContentIndex int
	Partial      AssistantMessage
}

// ThinkingDeltaEvent carries an incremental thinking chunk.
type ThinkingDeltaEvent struct {
	ContentIndex int
	Delta        string
	Partial      AssistantMessage
}

// ThinkingEndEvent marks the end of a thinking content block.
type ThinkingEndEvent struct {
	ContentIndex int
	Content      string
	Partial      AssistantMessage
}

// ToolCallStartEvent marks the beginning of a tool-call content block.
type ToolCallStartEvent struct {
	ContentIndex int
	Partial      AssistantMessage
}

// ToolCallDeltaEvent carries an incremental tool-call argument JSON chunk.
type ToolCallDeltaEvent struct {
	ContentIndex int
	Delta        string
	Partial      AssistantMessage
}

// ToolCallEndEvent marks the end of a tool-call content block.
type ToolCallEndEvent struct {
	ContentIndex int
	ToolCall     ToolCall
	Partial      AssistantMessage
}

// DoneEvent terminates the stream on success.
type DoneEvent struct {
	Reason  StopReason // "stop" | "length" | "toolUse"
	Message AssistantMessage
}

// ErrorEvent terminates the stream on failure.
type ErrorEvent struct {
	Reason StopReason // "aborted" | "error"
	Error  AssistantMessage
}

func (StartEvent) assistantMessageEvent()        {}
func (TextStartEvent) assistantMessageEvent()    {}
func (TextDeltaEvent) assistantMessageEvent()    {}
func (TextEndEvent) assistantMessageEvent()      {}
func (ThinkingStartEvent) assistantMessageEvent() {}
func (ThinkingDeltaEvent) assistantMessageEvent() {}
func (ThinkingEndEvent) assistantMessageEvent()   {}
func (ToolCallStartEvent) assistantMessageEvent() {}
func (ToolCallDeltaEvent) assistantMessageEvent() {}
func (ToolCallEndEvent) assistantMessageEvent()   {}
func (DoneEvent) assistantMessageEvent()          {}
func (ErrorEvent) assistantMessageEvent()         {}

func (StartEvent) EventType() string        { return "start" }
func (TextStartEvent) EventType() string    { return "text_start" }
func (TextDeltaEvent) EventType() string    { return "text_delta" }
func (TextEndEvent) EventType() string      { return "text_end" }
func (ThinkingStartEvent) EventType() string { return "thinking_start" }
func (ThinkingDeltaEvent) EventType() string { return "thinking_delta" }
func (ThinkingEndEvent) EventType() string   { return "thinking_end" }
func (ToolCallStartEvent) EventType() string { return "toolcall_start" }
func (ToolCallDeltaEvent) EventType() string { return "toolcall_delta" }
func (ToolCallEndEvent) EventType() string   { return "toolcall_end" }
func (DoneEvent) EventType() string          { return "done" }
func (ErrorEvent) EventType() string         { return "error" }

// IsTerminal reports whether e is a done or error event (stream terminator).
func IsTerminal(e AssistantMessageEvent) bool {
	switch e.(type) {
	case DoneEvent, ErrorEvent:
		return true
	}
	return false
}

// TerminalMessage returns the final AssistantMessage carried by a terminal
// (done/error) event, or the zero value for non-terminal events.
func TerminalMessage(e AssistantMessageEvent) (AssistantMessage, bool) {
	switch ev := e.(type) {
	case DoneEvent:
		return ev.Message, true
	case ErrorEvent:
		return ev.Error, true
	}
	return AssistantMessage{}, false
}
