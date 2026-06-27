package main

import (
	"bufio"
	"os"
	"strings"
)

// loadDotEnv reads a .env file (if present) and sets env vars that are not
// already set in the process environment. Existing env vars take precedence, so
// a shell export or CI secret always wins over the file. This keeps secrets out
// of the committed code while remaining optional — no error if .env is absent.
func loadDotEnv(path string) {
	f, err := os.Open(path)
	if err != nil {
		return // missing .env is fine
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		// Strip surrounding quotes if present.
		if len(value) >= 2 && ((value[0] == '"' && value[len(value)-1] == '"') ||
			(value[0] == '\'' && value[len(value)-1] == '\'')) {
			value = value[1 : len(value)-1]
		}
		// Do not override an already-set env var.
		if _, present := os.LookupEnv(key); present {
			continue
		}
		_ = os.Setenv(key, value)
	}
}
