package permission

import "regexp"

// destructiveBashRe matches bash commands that are unconditionally destructive
// and should be denied (not merely asked) even in default mode.
var destructiveBashRe = regexp.MustCompile(`(?i)(\brm\s+-rf?\s+/(?:\s|$)|git\s+push\s+--force|git\s+push\s+-f\b|>\s*/dev/sd|mkfs|dd\s+.*of=|:\(\)\{.*\|\:&\};:|chmod\s+-R\s+777\s+/)`)

// builtinPolicy returns the built-in deny/ask rules applied before custom rules.
// Deny rules block outright; everything else falls through to mode/ask logic.
func builtinPolicy() []Rule {
	return []Rule{
		// Hard-deny obviously destructive bash commands.
		{
			Tool:       "bash",
			Kind:       "deny",
			ArgPattern: destructiveBashRe,
		},
	}
}

// SensitivePathPrefixes are path prefixes that file tools may never touch
// regardless of mode (unless a custom allow rule explicitly permits).
var SensitivePathPrefixes = []string{
	// Secrets
	".ssh",
	".env",
	".npmrc",
	".pypirc",
	// VCS internals
	".git",
}
