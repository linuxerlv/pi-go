package ai

import (
	"encoding/json"
	"fmt"
)

// This file adds JSON (un)marshaling for the content-block and message union
// types, which are interfaces in Go. Serialization is a tagged union on the
// Type / Role field, mirroring the wire shape providers already use.

// ---- Content blocks ----

// MarshalJSON encodes a ToolCall, ensuring arguments survive even when only
// Arguments (not ArgumentsRaw) is populated.
func (c ToolCall) MarshalJSON() ([]byte, error) {
	args := c.ArgumentsRaw
	if len(args) == 0 && len(c.Arguments) > 0 {
		b, err := json.Marshal(c.Arguments)
		if err != nil {
			return nil, err
		}
		args = b
	}
	return json.Marshal(struct {
		Type           string          `json:"type"`
		ID             string          `json:"id"`
		Name           string          `json:"name"`
		Arguments      json.RawMessage `json:"arguments,omitempty"`
		ThoughtSignature string        `json:"thoughtSignature,omitempty"`
	}{
		Type:             c.Type,
		ID:               c.ID,
		Name:             c.Name,
		Arguments:        json.RawMessage(args),
		ThoughtSignature: c.ThoughtSignature,
	})
}

// UnmarshalContentBlock decodes a JSON object into the right ContentBlock
// variant based on its "type" field.
func UnmarshalContentBlock(data []byte) (ContentBlock, error) {
	var probe struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, err
	}
	switch probe.Type {
	case "text":
		var b TextContent
		return b, json.Unmarshal(data, &b)
	case "thinking":
		var b ThinkingContent
		return b, json.Unmarshal(data, &b)
	case "image":
		var b ImageContent
		return b, json.Unmarshal(data, &b)
	case "toolCall":
		var b ToolCall
		if err := json.Unmarshal(data, &b); err != nil {
			return nil, err
		}
		if len(b.ArgumentsRaw) == 0 && len(b.Arguments) > 0 {
			if raw, err := json.Marshal(b.Arguments); err == nil {
				b.ArgumentsRaw = raw
			}
		}
		return b, nil
	}
	return nil, fmt.Errorf("unknown content block type: %q", probe.Type)
}

// contentBlockSlice supports (un)marshaling []ContentBlock.
type contentBlockSlice []ContentBlock

func (s contentBlockSlice) MarshalJSON() ([]byte, error) {
	out := make([]json.RawMessage, 0, len(s))
	for _, b := range s {
		raw, err := json.Marshal(b)
		if err != nil {
			return nil, err
		}
		out = append(out, raw)
	}
	return json.Marshal(out)
}

func (s *contentBlockSlice) UnmarshalJSON(data []byte) error {
	var raws []json.RawMessage
	if err := json.Unmarshal(data, &raws); err != nil {
		return err
	}
	*s = make([]ContentBlock, 0, len(raws))
	for _, raw := range raws {
		b, err := UnmarshalContentBlock(raw)
		if err != nil {
			return err
		}
		*s = append(*s, b)
	}
	return nil
}

// ---- Messages ----

// UnmarshalMessage decodes a JSON object into the right Message variant based
// on its "role" field.
func UnmarshalMessage(data []byte) (Message, error) {
	var probe struct {
		Role string `json:"role"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, err
	}
	switch probe.Role {
	case "user":
		var m rawUserMessage
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, err
		}
		return m.toUserMessage(), nil
	case "assistant":
		var m rawAssistantMessage
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, err
		}
		return m.toAssistantMessage(), nil
	case "toolResult":
		var m rawToolResultMessage
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, err
		}
		return m.toToolResultMessage(), nil
	}
	return nil, fmt.Errorf("unknown message role: %q", probe.Role)
}

// rawUserMessage is a JSON-friendly UserMessage: content is decoded into either
// a string or a []ContentBlock.
type rawUserMessage struct {
	Role      string          `json:"role"`
	Content   json.RawMessage `json:"content"`
	Timestamp int64           `json:"timestamp"`
}

func (r rawUserMessage) toUserMessage() UserMessage {
	m := UserMessage{Timestamp: r.Timestamp}
	// Try string first.
	var s string
	if err := json.Unmarshal(r.Content, &s); err == nil {
		m.Content = s
		return m
	}
	var blocks contentBlockSlice
	if err := json.Unmarshal(r.Content, &blocks); err == nil {
		m.Content = []ContentBlock(blocks)
	}
	return m
}

// rawAssistantMessage mirrors AssistantMessage but with a content block slice
// that can round-trip through JSON.
type rawAssistantMessage struct {
	Role          string            `json:"role"`
	Content       contentBlockSlice `json:"content"`
	API           Api               `json:"api"`
	Provider      string            `json:"provider"`
	Model         string            `json:"model"`
	ResponseModel string            `json:"responseModel,omitempty"`
	ResponseID    string            `json:"responseId,omitempty"`
	Usage         Usage             `json:"usage"`
	StopReason    StopReason        `json:"stopReason"`
	ErrorMessage  string            `json:"errorMessage,omitempty"`
	Timestamp     int64             `json:"timestamp"`
}

func (r rawAssistantMessage) toAssistantMessage() AssistantMessage {
	return AssistantMessage{
		Content:       []ContentBlock(r.Content),
		API:           r.API,
		Provider:      r.Provider,
		Model:         r.Model,
		ResponseModel: r.ResponseModel,
		ResponseID:    r.ResponseID,
		Usage:         r.Usage,
		StopReason:    r.StopReason,
		ErrorMessage:  r.ErrorMessage,
		Timestamp:     r.Timestamp,
	}
}

// rawToolResultMessage mirrors ToolResultMessage.
type rawToolResultMessage struct {
	Role       string            `json:"role"`
	ToolCallID string            `json:"toolCallId"`
	ToolName   string            `json:"toolName"`
	Content    contentBlockSlice `json:"content"`
	Details    json.RawMessage   `json:"details,omitempty"`
	IsError    bool              `json:"isError"`
	Timestamp  int64             `json:"timestamp"`
}

func (r rawToolResultMessage) toToolResultMessage() ToolResultMessage {
	m := ToolResultMessage{
		ToolCallID: r.ToolCallID,
		ToolName:   r.ToolName,
		Content:    []ContentBlock(r.Content),
		IsError:    r.IsError,
		Timestamp:  r.Timestamp,
	}
	if len(r.Details) > 0 && string(r.Details) != "null" {
		var v any
		if json.Unmarshal(r.Details, &v) == nil {
			m.Details = v
		}
	}
	return m
}

// MarshalMessage encodes a Message to JSON.
func MarshalMessage(m Message) ([]byte, error) {
	switch msg := m.(type) {
	case UserMessage:
		return marshalUserMessage(msg)
	case AssistantMessage:
		return marshalAssistantMessage(msg)
	case ToolResultMessage:
		return marshalToolResultMessage(msg)
	}
	return nil, fmt.Errorf("unsupported message type: %T", m)
}

func marshalUserMessage(m UserMessage) ([]byte, error) {
	var contentRaw json.RawMessage
	switch c := m.Content.(type) {
	case string:
		b, err := json.Marshal(c)
		if err != nil {
			return nil, err
		}
		contentRaw = b
	case []ContentBlock:
		b, err := json.Marshal(contentBlockSlice(c))
		if err != nil {
			return nil, err
		}
		contentRaw = b
	default:
		// Fallback: encode as-is.
		b, err := json.Marshal(c)
		if err != nil {
			return nil, err
		}
		contentRaw = b
	}
	return json.Marshal(struct {
		Role      string          `json:"role"`
		Content   json.RawMessage `json:"content"`
		Timestamp int64           `json:"timestamp"`
	}{"user", contentRaw, m.Timestamp})
}

func marshalAssistantMessage(m AssistantMessage) ([]byte, error) {
	r := rawAssistantMessage{
		Role:          "assistant",
		Content:       contentBlockSlice(m.Content),
		API:           m.API,
		Provider:      m.Provider,
		Model:         m.Model,
		ResponseModel: m.ResponseModel,
		ResponseID:    m.ResponseID,
		Usage:         m.Usage,
		StopReason:    m.StopReason,
		ErrorMessage:  m.ErrorMessage,
		Timestamp:     m.Timestamp,
	}
	return json.Marshal(r)
}

func marshalToolResultMessage(m ToolResultMessage) ([]byte, error) {
	r := rawToolResultMessage{
		Role:       "toolResult",
		ToolCallID: m.ToolCallID,
		ToolName:   m.ToolName,
		Content:    contentBlockSlice(m.Content),
		IsError:    m.IsError,
		Timestamp:  m.Timestamp,
	}
	if m.Details != nil {
		b, err := json.Marshal(m.Details)
		if err != nil {
			return nil, err
		}
		r.Details = b
	}
	return json.Marshal(r)
}
