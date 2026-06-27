package main

import (
	"os"
	"strings"
)

// osStderr returns os.Stderr (indirected so tests could swap it).
func osStderr() *os.File { return os.Stderr }

// isTTY reports whether stdout is a terminal (for ANSI coloring decisions).
func isTTY() bool {
	fi, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return (fi.Mode() & os.ModeCharDevice) != 0
}

// dim wraps s in ANSI dim styling, stripped if stdout is not a TTY.
func dim(s string) string {
	if !isTTY() {
		return s
	}
	return "\033[2m" + s + "\033[0m"
}

// red wraps s in ANSI red, stripped if stdout is not a TTY.
func red(s string) string {
	if !isTTY() {
		return s
	}
	return "\033[31m" + s + "\033[0m"
}

// stringsReplaceAll is a thin alias to strings.ReplaceAll for short call sites.
func stringsReplaceAll(s, old, new string) string {
	return strings.ReplaceAll(s, old, new)
}
