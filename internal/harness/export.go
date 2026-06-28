package harness

import (
	"fmt"
	"html"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/linuxerlv/pi-go/internal/ai"
)

// ExportHTML writes the session's conversation as a self-contained HTML file.
// ANSI colors are stripped (text-only), and messages are rendered with role-
// based styling. Mirrors pi-coding-agent's export-html in a minimal form.
func ExportHTML(sess *Session, path string) error {
	ctx := sess.BuildContext()
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	io.WriteString(f, htmlHeader(sess.Metadata().ID))
	for _, m := range ctx.Messages {
		renderMessageHTML(f, m)
	}
	io.WriteString(f, htmlFooter)
	return nil
}

// htmlHeader returns the HTML document header for a session export.
func htmlHeader(sessionID string) string {
	return fmt.Sprintf(`<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>pi-go session %s</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;color:#222}
.msg{border-left:3px solid #ccc;margin:1em 0;padding:0.5em 1em;background:#fafafa}
.msg.user{border-color:#39c}
.msg.assistant{border-color:#63f}
.msg.tool{border-color:#d6a;border-left-style:dashed;background:#f6f6f6}
.role{font-size:0.8em;color:#888;text-transform:uppercase;margin-bottom:0.3em}
pre{white-space:pre-wrap;word-wrap:break-word;margin:0;font-family:ui-monospace,Consolas,monospace;font-size:0.9em}
</style></head><body>
<h2>pi-go session: %s</h2>
`, sessionID, sessionID)
}

func renderMessageHTML(w io.Writer, m ai.Message) {
	switch msg := m.(type) {
	case ai.UserMessage:
		text := userText(msg)
		fmt.Fprintf(w, `<div class="msg user"><div class="role">user</div><pre>%s</pre></div>`+"\n", html.EscapeString(text))
	case ai.AssistantMessage:
		var sb strings.Builder
		for _, b := range msg.Content {
			if t, ok := b.(ai.TextContent); ok {
				sb.WriteString(t.Text)
			}
		}
		fmt.Fprintf(w, `<div class="msg assistant"><div class="role">assistant</div><pre>%s</pre></div>`+"\n", html.EscapeString(sb.String()))
	case ai.ToolResultMessage:
		var sb strings.Builder
		for _, b := range msg.Content {
			if t, ok := b.(ai.TextContent); ok {
				sb.WriteString(t.Text)
			}
		}
		fmt.Fprintf(w, `<div class="msg tool"><div class="role">tool: %s</div><pre>%s</pre></div>`+"\n", html.EscapeString(msg.ToolName), html.EscapeString(sb.String()))
	}
}

func userText(m ai.UserMessage) string {
	switch c := m.Content.(type) {
	case string:
		return c
	case []ai.ContentBlock:
		var parts []string
		for _, b := range c {
			if t, ok := b.(ai.TextContent); ok {
				parts = append(parts, t.Text)
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

// ExportJSONL copies a session's jsonl file to path. The session's own jsonl
// is the canonical form, so this is a file copy.
func ExportJSONL(sess *Session, path string) error {
	src := sess.Storage()
	js, ok := src.(*JsonlStorage)
	if !ok {
		return fmt.Errorf("export jsonl requires a JsonlStorage session")
	}
	in, err := os.Open(js.entryPath())
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	return err
}

// ImportJSONL copies a jsonl file into the session directory under the given id,
// returning a Session over the imported data.
func ImportJSONL(mgr *SessionManager, srcPath, id string) (*Session, error) {
	if id == "" {
		id = "imported-" + nowCompact()
	}
	dst := filepath.Join(mgr.dir, id+".jsonl")
	in, err := os.Open(srcPath)
	if err != nil {
		return nil, err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return nil, err
	}
	defer out.Close()
	if _, err := io.Copy(out, in); err != nil {
		return nil, err
	}
	return mgr.Open(id)
}

var htmlFooter = "</body></html>\n"
