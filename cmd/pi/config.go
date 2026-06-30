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

// defaultConfigPath returns the project-level config file path.
func defaultConfigPath() string { return filepath.Join(".pi-go", "config.json") }

// userConfigPath returns the user-level config file path.
// Uses os.UserConfigDir for cross-platform behaviour:
//   Linux/macOS: $XDG_CONFIG_HOME/pi-go/config.json (default ~/.config/pi-go/config.json)
//   Windows:     %APPDATA%/pi-go/config.json
func userConfigPath() string {
	dir, err := os.UserConfigDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "pi-go", "config.json")
}

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

// mergeConfigs returns a new Config where any non-zero field in override
// replaces the corresponding field in base.
func mergeConfigs(base, override Config) Config {
	if override.Provider != "" {
		base.Provider = override.Provider
	}
	if override.Model != "" {
		base.Model = override.Model
	}
	if override.BaseURL != "" {
		base.BaseURL = override.BaseURL
	}
	if override.SystemPrompt != "" {
		base.SystemPrompt = override.SystemPrompt
	}
	return base
}

// loadMergedConfig loads config with priority: project > user.
// Callers apply env vars and CLI flags on top of the returned Config.
func loadMergedConfig() Config {
	cfg := loadConfig(userConfigPath())
	cfg = mergeConfigs(cfg, loadConfig(defaultConfigPath()))
	return cfg
}
