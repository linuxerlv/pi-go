package harness

import (
	"fmt"
	"strings"
)

// resources holds skills and prompt templates plus the system-prompt assembly
// logic. It is an internal component of AgentHarness, extracted to isolate the
// "available resources" concern from the run loop.
type resources struct {
	skills    []Skill
	templates []PromptTemplate
}

// effectiveSystemPrompt returns the base system prompt with a list of available
// (non-hidden) skills appended.
func (r *resources) effectiveSystemPrompt(base string) string {
	var visible []Skill
	for _, s := range r.skills {
		if !s.DisableModelInvocation {
			visible = append(visible, s)
		}
	}
	if len(visible) == 0 {
		return base
	}
	var sb strings.Builder
	if base != "" {
		sb.WriteString(base)
		sb.WriteString("\n\n")
	}
	sb.WriteString("Available skills (invoke via the application's skill command):\n")
	for _, s := range visible {
		sb.WriteString(fmt.Sprintf("- %s", s.Name))
		if s.Description != "" {
			sb.WriteString(": ")
			sb.WriteString(s.Description)
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

func (r *resources) findSkill(name string) (Skill, bool) {
	for _, s := range r.skills {
		if s.Name == name {
			return s, true
		}
	}
	return Skill{}, false
}

func (r *resources) findTemplate(name string) (PromptTemplate, bool) {
	for _, t := range r.templates {
		if t.Name == name {
			return t, true
		}
	}
	return PromptTemplate{}, false
}
