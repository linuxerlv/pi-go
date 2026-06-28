package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"
)

// makeStdinAsker returns a permission.Asker that prompts on stdin/stderr for
// the line-mode (non-TUI) REPL. It shows the operation and reads y/n/a
// (yes / no / always-allow).
func makeStdinAsker() func(ctx context.Context, prompt string) (string, error) {
	reader := bufio.NewReader(os.Stdin)
	return func(ctx context.Context, prompt string) (string, error) {
		fmt.Fprintf(os.Stderr, "\n\033[33mpermission required\033[0m: %s\n", prompt)
		fmt.Fprint(os.Stderr, "allow? [y]es / [n]o / [a]lways-allow: ")
		line, err := reader.ReadString('\n')
		if err != nil {
			return "deny", err
		}
		switch strings.TrimSpace(strings.ToLower(line)) {
		case "y", "yes":
			return "allow", nil
		case "a", "always", "always-allow":
			return "allow-always", nil
		default:
			return "deny", nil
		}
	}
}
