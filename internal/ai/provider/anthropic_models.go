package provider

import "github.com/linuxerlv/pi-go/internal/ai"

// builtinAnthropicModels returns a small catalog of current Anthropic models.
// Costs are USD per million tokens (input/output). Context/max-tokens values
// follow the public model card; they are advisory, not enforced.
func builtinAnthropicModels() []ai.Model {
	return []ai.Model{
		{
			ID:            "claude-opus-4-6",
			Name:          "Claude Opus 4.6",
			API:           ai.APIAnthropicMessages,
			Provider:      "anthropic",
			Reasoning:     true,
			Input:         []string{"text", "image"},
			ContextWindow: 200_000,
			MaxTokens:     32_000,
			Cost: ai.ModelCost{
				Input: 15, Output: 75, CacheRead: 1.5, CacheWrite: 18.75,
			},
		},
		{
			ID:            "claude-sonnet-4-6",
			Name:          "Claude Sonnet 4.6",
			API:           ai.APIAnthropicMessages,
			Provider:      "anthropic",
			Reasoning:     true,
			Input:         []string{"text", "image"},
			ContextWindow: 200_000,
			MaxTokens:     64_000,
			Cost: ai.ModelCost{
				Input: 3, Output: 15, CacheRead: 0.3, CacheWrite: 3.75,
			},
		},
		{
			ID:            "claude-haiku-4-5",
			Name:          "Claude Haiku 4.5",
			API:           ai.APIAnthropicMessages,
			Provider:      "anthropic",
			Reasoning:     true,
			Input:         []string{"text", "image"},
			ContextWindow: 200_000,
			MaxTokens:     64_000,
			Cost: ai.ModelCost{
				Input: 1, Output: 5, CacheRead: 0.1, CacheWrite: 1.25,
			},
		},
	}
}
