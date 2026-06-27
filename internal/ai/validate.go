package ai

import (
	"fmt"
	"strconv"
)

// ValidateToolArguments validates a tool call's arguments against the tool's
// JSON Schema and returns the (possibly coerced) arguments. It mirrors
// @earendil-works/pi-ai's validateToolArguments.
//
// This is a lightweight implementation covering the common cases used by the
// built-in tools: type coercion for primitive properties (string/number/
// integer/boolean) and required-field checking. Full JSON Schema (allOf/anyOf/
// oneOf/pattern/etc.) can be layered in later via a third-party validator.
func ValidateToolArguments(tool Tool, toolCall ToolCall) (map[string]any, error) {
	args := cloneMap(toolCall.Arguments)
	schema := tool.Parameters

	if schema != nil {
		args = coerceWithSchema(args, schema).(map[string]any)
		if err := validateObject(args, schema, ""); err != nil {
			return nil, fmt.Errorf(
				"Validation failed for tool %q:\n  - %s\n\nReceived arguments:\n%s",
				toolCall.Name, err.Error(), MustJSON(args),
			)
		}
	}
	if args == nil {
		args = map[string]any{}
	}
	return args, nil
}

func validateObject(value map[string]any, schema map[string]any, path string) error {
	props, _ := schema["properties"].(map[string]any)

	if required, ok := schema["required"].([]any); ok {
		for _, r := range required {
			name, _ := r.(string)
			if name == "" {
				continue
			}
			if _, present := value[name]; !present {
				p := name
				if path != "" {
					p = path + "." + name
				}
				return fmt.Errorf("%s: missing required property", p)
			}
		}
	}

	for key, propSchemaAny := range props {
		raw, present := value[key]
		if !present {
			continue
		}
		p := key
		if path != "" {
			p = path + "." + key
		}
		propSchema, ok := propSchemaAny.(map[string]any)
		if !ok {
			continue
		}
		if err := validateValue(raw, propSchema, p); err != nil {
			return err
		}
	}
	return nil
}

func validateValue(value any, schema map[string]any, path string) error {
	types := schemaTypes(schema)
	if len(types) == 0 {
		return nil
	}
	matched := false
	for _, t := range types {
		if matchesJSONType(value, t) {
			matched = true
			break
		}
	}
	if !matched {
		return fmt.Errorf("%s: expected type %v", path, types)
	}

	if containsString(types, "object") {
		if obj, ok := value.(map[string]any); ok {
			return validateObject(obj, schema, path)
		}
	}
	if containsString(types, "array") {
		if arr, ok := value.([]any); ok {
			itemsSchema, _ := schema["items"].(map[string]any)
			if itemsSchema != nil {
				for i, item := range arr {
					if err := validateValue(item, itemsSchema, fmt.Sprintf("%s[%d]", path, i)); err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}

// coerceWithSchema applies primitive type coercion based on the schema's
// declared type(s), mirroring pi's coerceWithJsonSchema for the common cases.
func coerceWithSchema(value any, schema map[string]any) any {
	if schema == nil {
		return value
	}
	types := schemaTypes(schema)
	if len(types) == 0 {
		return value
	}

	// If the value already matches one of the declared types, recurse into
	// object/array children only.
	for _, t := range types {
		if matchesJSONType(value, t) {
			switch t {
			case "object":
				if obj, ok := value.(map[string]any); ok {
					return coerceObject(obj, schema)
				}
			case "array":
				if arr, ok := value.([]any); ok {
					return coerceArray(arr, schema)
				}
			}
			return value
		}
	}

	// Otherwise attempt primitive coercion to the first matching declared type.
	for _, t := range types {
		if c, ok := coercePrimitive(value, t); ok {
			return c
		}
	}
	return value
}

func coerceObject(value map[string]any, schema map[string]any) map[string]any {
	props, _ := schema["properties"].(map[string]any)
	for key, propAny := range props {
		raw, ok := value[key]
		if !ok {
			continue
		}
		propSchema, ok := propAny.(map[string]any)
		if !ok {
			continue
		}
		value[key] = coerceWithSchema(raw, propSchema)
	}
	return value
}

func coerceArray(value []any, schema map[string]any) []any {
	itemsSchema, _ := schema["items"].(map[string]any)
	if itemsSchema == nil {
		return value
	}
	for i, item := range value {
		value[i] = coerceWithSchema(item, itemsSchema)
	}
	return value
}

func coercePrimitive(value any, t string) (any, bool) {
	switch t {
	case "number":
		switch v := value.(type) {
		case string:
			if f, err := strconv.ParseFloat(v, 64); err == nil {
				return f, true
			}
		case bool:
			if v {
				return float64(1), true
			}
			return float64(0), true
		}
	case "integer":
		switch v := value.(type) {
		case string:
			if i, err := strconv.ParseInt(v, 10, 64); err == nil {
				return i, true
			}
		case bool:
			if v {
				return int64(1), true
			}
			return int64(0), true
		}
	case "boolean":
		switch v := value.(type) {
		case string:
			if v == "true" {
				return true, true
			}
			if v == "false" {
				return false, true
			}
		case float64:
			if v == 1 {
				return true, true
			}
			if v == 0 {
				return false, true
			}
		}
	case "string":
		switch v := value.(type) {
		case float64:
			return strconv.FormatFloat(v, 'f', -1, 64), true
		case bool:
			return strconv.FormatBool(v), true
		case nil:
			return "", true
		}
	case "null":
		return nil, true
	}
	return value, false
}

func schemaTypes(schema map[string]any) []string {
	switch t := schema["type"].(type) {
	case string:
		return []string{t}
	case []any:
		out := make([]string, 0, len(t))
		for _, x := range t {
			if s, ok := x.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

func matchesJSONType(value any, t string) bool {
	switch t {
	case "string":
		_, ok := value.(string)
		return ok
	case "number":
		switch value.(type) {
		case float64, float32, int, int64, int32:
			return true
		}
		return false
	case "integer":
		switch v := value.(type) {
		case int, int32, int64:
			return true
		case float64:
			return v == float64(int64(v))
		case float32:
			return v == float32(int32(v))
		}
		return false
	case "boolean":
		_, ok := value.(bool)
		return ok
	case "null":
		return value == nil
	case "array":
		_, ok := value.([]any)
		return ok
	case "object":
		_, ok := value.(map[string]any)
		return ok
	}
	return false
}

func containsString(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

func cloneMap(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = cloneValue(v)
	}
	return out
}

func cloneValue(v any) any {
	switch x := v.(type) {
	case map[string]any:
		return cloneMap(x)
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			out[i] = cloneValue(item)
		}
		return out
	default:
		return v
	}
}
