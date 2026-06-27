package provider

import "github.com/linuxerlv/pi-go/internal/ai"

// builtinOpenAIResponsesModels returns models typically available via the
// Responses API. Advisory only.
func builtinOpenAIResponsesModels() []ai.Model {
	return []ai.Model{
		{
			ID:            "gpt-4o",
			Name:          "GPT-4o (Responses)",
			API:           ai.APIOpenAIResponses,
			Provider:      "openai-responses",
			Input:         []string{"text", "image"},
			ContextWindow: 128_000,
			MaxTokens:     16_384,
			Cost: ai.ModelCost{Input: 2.5, Output: 10, CacheRead: 1.25},
		},
		{
			ID:            "gpt-4o-mini",
			Name:          "GPT-4o mini (Responses)",
			API:           ai.APIOpenAIResponses,
			Provider:      "openai-responses",
			Input:         []string{"text", "image"},
			ContextWindow: 128_000,
			MaxTokens:     16_384,
			Cost: ai.ModelCost{Input: 0.15, Output: 0.6, CacheRead: 0.075},
		},
	}
}
