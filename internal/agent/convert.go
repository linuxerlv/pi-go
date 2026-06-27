package agent

import "github.com/linuxerlv/pi-go/internal/ai"

// DefaultConvertToLlm converts AgentMessage[] to ai.Message[] by passing through
// the base LLM message types (UserMessage/AssistantMessage/ToolResultMessage).
// Custom non-LLM messages are dropped. This mirrors pi-agent-core's default
// convertToLlm behavior.
func DefaultConvertToLlm(messages []AgentMessage) []ai.Message {
	out := make([]ai.Message, 0, len(messages))
	for _, m := range messages {
		if msg, ok := m.(ai.Message); ok {
			out = append(out, msg)
		}
	}
	return out
}
