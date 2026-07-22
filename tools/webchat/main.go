// webchat - a browser testing frontend for the pure-C transformer engine.
//
// Serves a single embedded page and a WebSocket endpoint; each request is
// bridged to the transformer CLI (and, for the real pretrained models, the
// Python tokenizer wrappers) as a subprocess run with cwd = repo root.
package main

import (
	_ "embed"
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
)

//go:embed index.html
var indexHTML []byte

type config struct {
	root     string
	exe      string
	python   string
	imagegen string // optional external text->image command (e.g. diffusers)
}

var cfg config

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	root := flag.String("root", "", "repo root (holds transformer + model/); default: auto-detect")
	exe := flag.String("exe", "", "transformer binary (default: next to webchat, then <root>)")
	python := flag.String("python", "python", "python interpreter for the tokenizer bridges")
	imagegen := flag.String("imagegen", "", "external text->image command (prompt on stdin, PNG path appended), e.g. \"python tools/imagegen_diffusers.py\"")
	flag.Parse()

	cfg.root = resolveRoot(*root)
	cfg.python = *python
	cfg.imagegen = *imagegen
	if *exe != "" {
		cfg.exe = *exe
	} else {
		cfg.exe = findTransformer(cfg.root)
	}
	if _, err := os.Stat(cfg.exe); err != nil {
		log.Printf("WARNING: transformer binary not found (looked next to webchat and in %s) — build it first", cfg.root)
	}

	http.HandleFunc("/", handleIndex)
	http.HandleFunc("/ws", handleWS)
	log.Printf("webchat on http://localhost%s   (root=%s, exe=%s)", *addr, cfg.root, cfg.exe)
	log.Fatal(http.ListenAndServe(*addr, nil))
}

// resolveRoot finds the repo root (the folder holding model/ and the tools/). If
// -root is given it wins; otherwise auto-detect so a built webchat.exe just works
// whether run from the repo root or double-clicked in tools/webchat/.
func resolveRoot(flagVal string) string {
	if flagVal != "" {
		if a, err := filepath.Abs(flagVal); err == nil {
			return a
		}
	}
	if cwd, err := os.Getwd(); err == nil && hasModelDir(cwd) {
		return cwd
	}
	// webchat.exe lives in <root>/tools/webchat/, so climb two levels.
	if self, err := os.Executable(); err == nil {
		up := filepath.Dir(filepath.Dir(filepath.Dir(self)))
		if hasModelDir(up) {
			return up
		}
	}
	cwd, _ := os.Getwd()
	return cwd
}

func hasModelDir(dir string) bool {
	fi, err := os.Stat(filepath.Join(dir, "model"))
	return err == nil && fi.IsDir()
}

// findTransformer locates the engine: next to the webchat binary first (so both
// exes can sit in one folder), then in root, then cwd.
func findTransformer(root string) string {
	name := "transformer"
	if runtime.GOOS == "windows" {
		name = "transformer.exe"
	}
	var candidates []string
	if self, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Join(filepath.Dir(self), name))
	}
	candidates = append(candidates, filepath.Join(root, name))
	if cwd, err := os.Getwd(); err == nil {
		candidates = append(candidates, filepath.Join(cwd, name))
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return filepath.Join(root, name) // default (may not exist -> warning above)
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(indexHTML)
}

func handleWS(w http.ResponseWriter, r *http.Request) {
	c, err := wsUpgrade(w, r)
	if err != nil {
		log.Printf("ws upgrade failed: %v", err)
		return
	}
	defer c.close()
	for {
		op, data, err := c.readMessage()
		if err != nil {
			return
		}
		if op != opText {
			continue
		}
		var msg map[string]any
		if err := json.Unmarshal(data, &msg); err != nil {
			sendJSON(c, map[string]any{"op": "error", "text": "invalid JSON message"})
			continue
		}
		handleMessage(c, msg)
	}
}

func handleMessage(c *wsConn, msg map[string]any) {
	switch op, _ := msg["op"].(string); op {
	case "list_models":
		sendJSON(c, map[string]any{"op": "models", "items": detectModels(cfg.root)})
	case "chat":
		handleChat(c, msg)
	case "imggen":
		handleImggen(c, msg)
	case "audgen":
		handleAudgen(c, msg)
	case "caption":
		handleCaption(c, msg)
	case "text2media":
		handleText2media(c, msg)
	default:
		sendJSON(c, map[string]any{"op": "error", "text": "unknown op: " + op})
	}
}

func sendJSON(c *wsConn, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		return
	}
	c.writeText(b)
}
