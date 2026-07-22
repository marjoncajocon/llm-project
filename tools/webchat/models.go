// models.go - Detect the models under <root>/model/ and run text generation
// against them via the transformer CLI (pure C) or the Python tokenizer bridges.
package main

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// runTimeout bounds any single subprocess (generation can legitimately take tens
// of seconds on long sequences, but should never run for minutes).
const runTimeout = 4 * time.Minute

// Model describes one model/<name>/ folder and how to chat with it.
type Model struct {
	Name        string `json:"name"`
	Family      string `json:"family"` // GPTC / LLMA / LLMQ
	Mode        string `json:"mode"`   // purec-text / gpt2 / llama
	Ckpt        string `json:"ckpt"`   // checkpoint path, relative to root
	HFRepo      string `json:"hf_repo,omitempty"`
	NeedsPython bool   `json:"needs_python"`
	Label       string `json:"label"`
}

// Known HuggingFace tokenizer repos for the real LLaMA-family models shipped here.
// Overridable per model via model/<name>/hf_tokenizer.txt.
var hfRepos = map[string]string{
	"smollm": "HuggingFaceTB/SmolLM2-135M",
	"qwen":   "Qwen/Qwen2.5-0.5B",
}

func detectModels(root string) []Model {
	var out []Model
	entries, err := os.ReadDir(filepath.Join(root, "model"))
	if err != nil {
		return out
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		if m := classify(root, e.Name()); m != nil {
			out = append(out, *m)
		}
	}
	return out
}

func findModel(root, name string) *Model {
	for _, m := range detectModels(root) {
		if m.Name == name {
			mm := m
			return &mm
		}
	}
	return nil
}

func classify(root, name string) *Model {
	folder := filepath.Join(root, "model", name)
	ckptRel := ""
	// Prefer the int8 (LLMQ) checkpoint when present: it streams 4x less memory
	// and its decode matmul is threaded, so it's faster than the f32 path.
	for _, cand := range []string{"model_int8.bin", "model.bin"} {
		if _, err := os.Stat(filepath.Join(folder, cand)); err == nil {
			ckptRel = "model/" + name + "/" + cand
			break
		}
	}
	if ckptRel == "" {
		return nil
	}
	magic := peekMagic(filepath.Join(root, filepath.FromSlash(ckptRel)))
	_, hasTok := statOK(filepath.Join(folder, "tok.bin"))

	m := &Model{Name: name, Ckpt: ckptRel}
	switch magic {
	case "GPTC":
		m.Family = "GPTC"
		if hasTok {
			m.Mode = "purec-text"
			m.Label = name + " — trained BPE (pure C)"
		} else {
			m.Mode = "gpt2"
			m.NeedsPython = true
			m.Label = name + " — GPT-2 (needs Python tiktoken)"
		}
	case "LLMA", "LLMQ":
		m.Family = magic
		m.Mode = "llama"
		m.NeedsPython = true
		m.HFRepo = hfRepoFor(folder, name)
		q := ""
		if magic == "LLMQ" {
			q = " int8"
		}
		m.Label = name + " — LLaMA-family" + q + " (needs Python tokenizers)"
	default:
		return nil
	}
	return m
}

func peekMagic(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	var b [4]byte
	if _, err := io.ReadFull(f, b[:]); err != nil {
		return ""
	}
	return string(b[:])
}

func hfRepoFor(folder, name string) string {
	if data, err := os.ReadFile(filepath.Join(folder, "hf_tokenizer.txt")); err == nil {
		if s := strings.TrimSpace(string(data)); s != "" {
			return s
		}
	}
	return hfRepos[name]
}

func statOK(path string) (os.FileInfo, bool) {
	fi, err := os.Stat(path)
	return fi, err == nil
}

// runChat generates a reply for `text` using model m. Returns (reply, statusText, err).
func runChat(m Model, text string, n int) (string, string, error) {
	switch m.Mode {
	case "purec-text":
		// generate --name X reads ONE stdin line and prints "<prompt><continuation>".
		// The continuation can contain newlines (the model emits them), so the
		// output spans multiple lines — keep ALL of it, not just the first line
		// (that truncation made replies look cut off after the first token).
		// Flatten the prompt to a single line so it stays one stdin line and the
		// echoed prefix matches for stripping.
		prompt := strings.NewReplacer("\r", " ", "\n", " ").Replace(text)
		out, se, err := runCmd(cfg.exe, []string{"generate", "--name", m.Name}, prompt+"\n")
		if err != nil {
			return "", se, err
		}
		reply := strings.TrimPrefix(strings.TrimRight(out, "\r\n"), prompt)
		return cleanText(reply), se, nil
	case "gpt2":
		out, se, err := runCmd(cfg.python, []string{"tools/gpt2_chat.py", text, "--name", m.Name}, "")
		return cleanText(out), se, err
	case "llama":
		args := []string{"tools/llama_chat.py", text, "--ckpt", m.Ckpt, "--n", strconv.Itoa(n)}
		if m.HFRepo != "" {
			args = append(args, "--model", m.HFRepo)
		}
		out, se, err := runCmd(cfg.python, args, "")
		return cleanText(out), se, err
	}
	return "", "", fmt.Errorf("unsupported model mode %q", m.Mode)
}

// runCmd runs a subprocess with cwd = repo root (so model/, tools/*.py, and
// relative checkpoint paths resolve as the CLIs expect). Media/large-model runs
// can be slow (the engine's attention is O(seq^2) on CPU), so a generous context
// timeout guards against a genuine runaway rather than a long-but-normal run.
func runCmd(bin string, args []string, stdin string) (string, string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), runTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Dir = cfg.root
	if stdin != "" {
		cmd.Stdin = strings.NewReader(stdin)
	}
	var so, se bytes.Buffer
	cmd.Stdout = &so
	cmd.Stderr = &se
	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		err = fmt.Errorf("subprocess timed out after %s (try fewer steps / smaller input)", runTimeout)
	}
	return so.String(), se.String(), err
}

// cleanText normalizes Windows CRLF to LF and trims trailing blank lines.
func cleanText(s string) string {
	return strings.TrimRight(strings.ReplaceAll(s, "\r\n", "\n"), "\n ")
}

// runCmdStream runs a subprocess, feeds it stdin, and calls onLine for each
// stdout line as it arrives — used for live token streaming (the child flushes
// per token). Returns when the process exits.
func runCmdStream(bin string, args []string, stdin string, onLine func(string)) error {
	ctx, cancel := context.WithTimeout(context.Background(), runTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.Dir = cfg.root
	var errBuf bytes.Buffer
	cmd.Stderr = &errBuf
	inPipe, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	outPipe, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	go func() {
		io.WriteString(inPipe, stdin)
		inPipe.Close()
	}()
	sc := bufio.NewScanner(outPipe)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024) // allow long delta lines
	for sc.Scan() {
		onLine(sc.Text())
	}
	if err := cmd.Wait(); err != nil {
		if s := strings.TrimSpace(errBuf.String()); s != "" {
			return fmt.Errorf("%v: %s", err, lastLine(s))
		}
		return err
	}
	return nil
}

func lastLine(s string) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	return lines[len(lines)-1]
}

func firstNonEmptyLine(s string) string {
	for _, ln := range strings.Split(s, "\n") {
		ln = strings.TrimRight(ln, "\r")
		if strings.TrimSpace(ln) != "" {
			return ln
		}
	}
	return ""
}
