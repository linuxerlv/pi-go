// Package orchestrator implements multi-agent task decomposition and execution.
// It is a Go port of the high-level orchestration layer from
// @earendil-works/pi: given a complex task, it asks an LLM to break it into
// subtasks, runs each subtask through its own agent loop, and synthesizes the
// results into a final answer.
package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/linuxerlv/pi-go/internal/agent"
	"github.com/linuxerlv/pi-go/internal/ai"
)

// Strategy controls how a task is planned, executed, and synthesized.
type Strategy interface {
	// Plan breaks the overall task into subtasks.
	Plan(ctx context.Context, task string, o *Orchestrator) ([]SubTask, error)
	// Execute runs the subtasks and returns their individual results.
	Execute(ctx context.Context, subtasks []SubTask, o *Orchestrator) ([]SubTaskResult, error)
	// Synthesize combines subtask results into a final answer.
	Synthesize(ctx context.Context, task string, results []SubTaskResult, o *Orchestrator) (string, error)
}

// SubTask is a single unit of work produced by a Strategy.
type SubTask struct {
	ID          string `json:"id"`
	Description string `json:"description"`
}

// SubTaskResult is the output of executing one SubTask.
type SubTaskResult struct {
	SubTask SubTask
	Output  string
}

// Result is the final output of an Orchestrator run.
type Result struct {
	Task       string
	SubTasks   []SubTask
	Results    []SubTaskResult
	FinalAnswer string
}

// Options configures an Orchestrator.
type Options struct {
	Provider     ai.Provider
	Model        ai.Model
	SystemPrompt string
	Tools        []agent.AgentTool
	Strategy     Strategy
	MaxTokens    int
}

// Orchestrator coordinates multiple agent runs to solve a single high-level task.
type Orchestrator struct {
	provider     ai.Provider
	model        ai.Model
	systemPrompt string
	tools        []agent.AgentTool
	strategy     Strategy
	maxTokens    int
}

// New creates an Orchestrator. If opts.Strategy is nil, a sequential strategy is
// used by default.
func New(opts Options) *Orchestrator {
	maxTokens := opts.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}
	strategy := opts.Strategy
	if strategy == nil {
		strategy = &SequentialStrategy{}
	}
	return &Orchestrator{
		provider:     opts.Provider,
		model:        opts.Model,
		systemPrompt: opts.SystemPrompt,
		tools:        opts.Tools,
		strategy:     strategy,
		maxTokens:    maxTokens,
	}
}

// Run executes the task by planning, executing, and synthesizing.
func (o *Orchestrator) Run(ctx context.Context, task string) (*Result, error) {
	subtasks, err := o.strategy.Plan(ctx, task, o)
	if err != nil {
		return nil, fmt.Errorf("plan failed: %w", err)
	}
	if len(subtasks) == 0 {
		return nil, fmt.Errorf("strategy returned no subtasks")
	}

	results, err := o.strategy.Execute(ctx, subtasks, o)
	if err != nil {
		return nil, fmt.Errorf("execute failed: %w", err)
	}

	final, err := o.strategy.Synthesize(ctx, task, results, o)
	if err != nil {
		return nil, fmt.Errorf("synthesize failed: %w", err)
	}

	return &Result{
		Task:        task,
		SubTasks:    subtasks,
		Results:     results,
		FinalAnswer: final,
	}, nil
}

// RunSubtask runs a single subtask through a fresh agent loop and returns the
// plain-text result. It is the default execution primitive used by strategies.
func (o *Orchestrator) RunSubtask(ctx context.Context, subtask SubTask) (string, error) {
	ctx_ := agent.AgentContext{
		SystemPrompt: o.systemPrompt,
		Tools:        o.tools,
	}
	prompts := []agent.AgentMessage{
		ai.UserMessage{Content: subtask.Description, Timestamp: ai.Now()},
	}
	config := agent.NewLoopConfig(o.model).
		WithConvertToLlm(agent.DefaultConvertToLlm).
		Build()

	var parts []string
	emit := func(ev agent.AgentEvent) error {
		if me, ok := ev.(agent.MessageEndEvent); ok {
			if am, ok := me.Message.(ai.AssistantMessage); ok {
				parts = append(parts, assistantText(am))
			}
		}
		return nil
	}

	if _, err := agent.RunAgentLoop(ctx, prompts, ctx_, config, o.provider, emit); err != nil {
		return "", err
	}
	return strings.TrimSpace(strings.Join(parts, "")), nil
}

// streamText sends a single-turn prompt to the LLM and returns the text response.
func (o *Orchestrator) streamText(ctx context.Context, systemPrompt, userPrompt string, maxTokens int) (string, error) {
	if maxTokens == 0 {
		maxTokens = o.maxTokens
	}
	llmCtx := ai.Context{
		SystemPrompt: systemPrompt,
		Messages:     []ai.Message{ai.UserMessage{Content: userPrompt, Timestamp: ai.Now()}},
	}
	stream := o.provider.StreamSimple(ctx, o.model, llmCtx, &ai.SimpleStreamOptions{
		StreamOptions: ai.StreamOptions{MaxTokens: maxTokens},
	})

	var parts []string
	for ev := range stream.Range {
		switch e := ev.(type) {
		case ai.TextDeltaEvent:
			parts = append(parts, e.Delta)
		case ai.ErrorEvent:
			if final, _ := ai.TerminalMessage(e); final.ErrorMessage != "" {
				return "", fmt.Errorf("stream error: %s", final.ErrorMessage)
			}
		}
	}
	return strings.TrimSpace(strings.Join(parts, "")), nil
}

// SequentialStrategy plans subtasks, runs them one by one, and synthesizes the
// final answer. It is the safest default when subtasks depend on each other.
type SequentialStrategy struct{}

func (s *SequentialStrategy) Plan(ctx context.Context, task string, o *Orchestrator) ([]SubTask, error) {
	const systemPrompt = "You are a task planner. Break the user's task into a JSON array of subtasks. Each subtask must have an id and a description. Be concise."
	userPrompt := fmt.Sprintf("Break this task into subtasks and respond with JSON only (no markdown, no prose):\n\n%s", task)
	text, err := o.streamText(ctx, systemPrompt, userPrompt, 2048)
	if err != nil {
		return nil, err
	}
	return parseSubTasks(text)
}

func (s *SequentialStrategy) Execute(ctx context.Context, subtasks []SubTask, o *Orchestrator) ([]SubTaskResult, error) {
	out := make([]SubTaskResult, 0, len(subtasks))
	for _, st := range subtasks {
		output, err := o.RunSubtask(ctx, st)
		if err != nil {
			return out, fmt.Errorf("subtask %s failed: %w", st.ID, err)
		}
		out = append(out, SubTaskResult{SubTask: st, Output: output})
	}
	return out, nil
}

func (s *SequentialStrategy) Synthesize(ctx context.Context, task string, results []SubTaskResult, o *Orchestrator) (string, error) {
	const systemPrompt = "You are a helpful assistant. Synthesize the subtask results into a final answer for the user's original task."
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Original task: %s\n\nSubtask results:\n", task))
	for _, r := range results {
		sb.WriteString(fmt.Sprintf("- %s: %s\n", r.SubTask.ID, r.Output))
	}
	sb.WriteString("\nFinal answer:")
	return o.streamText(ctx, systemPrompt, sb.String(), 4096)
}

// ParallelStrategy plans subtasks and runs them concurrently. It is useful when
// subtasks are independent (e.g. researching multiple topics).
type ParallelStrategy struct {
	// MaxConcurrency limits how many subtasks run at once. Zero means no limit.
	MaxConcurrency int
}

func (s *ParallelStrategy) Plan(ctx context.Context, task string, o *Orchestrator) ([]SubTask, error) {
	// Reuse the same planner as sequential.
	seq := &SequentialStrategy{}
	return seq.Plan(ctx, task, o)
}

func (s *ParallelStrategy) Execute(ctx context.Context, subtasks []SubTask, o *Orchestrator) ([]SubTaskResult, error) {
	limit := s.MaxConcurrency
	if limit <= 0 {
		limit = len(subtasks)
	}
	sem := make(chan struct{}, limit)
	var wg sync.WaitGroup
	results := make([]SubTaskResult, len(subtasks))
	var errOnce sync.Once
	var execErr error

	for i, st := range subtasks {
		wg.Add(1)
		go func(idx int, sub SubTask) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			output, err := o.RunSubtask(ctx, sub)
			if err != nil {
				errOnce.Do(func() { execErr = fmt.Errorf("subtask %s failed: %w", sub.ID, err) })
				// Record a placeholder so the synthesis phase sees the failure
				// rather than an empty zero-value result.
				results[idx] = SubTaskResult{SubTask: sub, Output: fmt.Sprintf("[subtask %s failed: %v]", sub.ID, err)}
				return
			}
			results[idx] = SubTaskResult{SubTask: sub, Output: output}
		}(i, st)
	}
	wg.Wait()
	if execErr != nil {
		return results, execErr
	}
	return results, nil
}

func (s *ParallelStrategy) Synthesize(ctx context.Context, task string, results []SubTaskResult, o *Orchestrator) (string, error) {
	seq := &SequentialStrategy{}
	return seq.Synthesize(ctx, task, results, o)
}

func parseSubTasks(text string) ([]SubTask, error) {
	// Strip markdown code fences if present.
	text = strings.TrimSpace(text)
	if idx := strings.Index(text, "["); idx >= 0 {
		if closeIdx := strings.LastIndex(text, "]"); closeIdx > idx {
			text = text[idx : closeIdx+1]
		}
	}
	var tasks []SubTask
	if err := json.Unmarshal([]byte(text), &tasks); err != nil {
		// Fallback: treat the whole response as a single subtask.
		return []SubTask{{ID: "1", Description: text}}, nil
	}
	// Normalize ids.
	for i := range tasks {
		if tasks[i].ID == "" {
			tasks[i].ID = fmt.Sprintf("%d", i+1)
		}
		if tasks[i].Description == "" {
			tasks[i].Description = tasks[i].ID
		}
	}
	return tasks, nil
}

func assistantText(am ai.AssistantMessage) string {
	var parts []string
	for _, c := range am.Content {
		if t, ok := c.(ai.TextContent); ok {
			parts = append(parts, t.Text)
		}
	}
	return strings.Join(parts, "")
}
