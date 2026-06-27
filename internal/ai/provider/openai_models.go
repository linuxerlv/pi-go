package provider

import "github.com/linuxerlv/pi-go/internal/ai"

// builtinOpenAIModels returns a small catalog of OpenAI models. These are
// advisory (costs/limits); the real values come from the provider.
func builtinOpenAIModels() []ai.Model {
	return []ai.Model{
		{
			ID:            "gpt-4o",
			Name:          "GPT-4o",
			API:           ai.APIOpenAICompletions,
			Provider:      "openai",
			Input:         []string{"text", "image"},
			ContextWindow: 128_000,
			MaxTokens:     16_384,
			Cost: ai.ModelCost{
				Input: 2.5, Output: 10, CacheRead: 1.25,
			},
		},
		{
			ID:            "gpt-4o-mini",
			Name:          "GPT-4o mini",
			API:           ai.APIOpenAICompletions,
			Provider:      "openai",
			Input:         []string{"text", "image"},
			ContextWindow: 128_000,
			MaxTokens:     16_384,
			Cost: ai.ModelCost{
				Input: 0.15, Output: 0.6, CacheRead: 0.075,
			},
		},
	}
}

// registerBuiltinOpenAIModels seeds an OpenAI provider's model table.
func registerBuiltinOpenAIModels(p *OpenAI) {
	for _, m := range builtinOpenAIModels() {
		p.RegisterModel(m)
	}
}
