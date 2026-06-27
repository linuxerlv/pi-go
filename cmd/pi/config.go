package main

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// Config holds optional CLI defaults loaded from .pi-go/config.json. Env vars
// and command-line flags always take precedence over this file.
type Config struct {
	Provider     string `json:"provider,omitempty"`
	Model        string `json:"model,omitempty"`
	BaseURL      string `json:"baseURL,omitempty"`
	SystemPrompt string `json:"systemPrompt,omitempty"`
}

// defaultConfigPath returns the config file path.
func defaultConfigPath() string { return filepath.Join(".pi-go", "config.json") }

// loadConfig reads the config file if present. Missing file is not an error.
func loadConfig(path string) Config {
	var c Config
	b, err := os.ReadFile(path)
	if err != nil {
		return c
	}
	_ = json.Unmarshal(b, &c)
	return c
}
