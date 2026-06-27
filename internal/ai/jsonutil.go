package ai

import (
	"encoding/json"
	"fmt"
)

// MustJSON marshals v to indented JSON, returning a fallback string on error.
// Intended for error-message and logging formatting where failure to marshal
// is not itself fatal.
func MustJSON(v any) string {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Sprintf("%+v", v)
	}
	return string(b)
}
