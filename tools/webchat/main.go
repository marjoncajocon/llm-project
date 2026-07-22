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
	root   string
	exe    string
	python string
}

var cfg config

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	root := flag.String("root", filepath.Join("..", ".."), "repo root (holds transformer + model/)")
	exe := flag.String("exe", "", "transformer binary (default <root>/transformer[.exe])")
	python := flag.String("python", "python", "python interpreter for the tokenizer bridges")
	flag.Parse()

	absRoot, err := filepath.Abs(*root)
	if err != nil {
		log.Fatalf("resolving root: %v", err)
	}
	cfg.root = absRoot
	cfg.python = *python
	if *exe != "" {
		cfg.exe = *exe
	} else {
		name := "transformer"
		if runtime.GOOS == "windows" {
			name = "transformer.exe"
		}
		cfg.exe = filepath.Join(absRoot, name)
	}
	if _, err := os.Stat(cfg.exe); err != nil {
		log.Printf("WARNING: transformer binary not found at %s — build it first", cfg.exe)
	}

	http.HandleFunc("/", handleIndex)
	http.HandleFunc("/ws", handleWS)
	log.Printf("webchat on http://localhost%s   (root=%s)", *addr, cfg.root)
	log.Fatal(http.ListenAndServe(*addr, nil))
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
	default:
		sendJSON(c, map[string]any{"op": "error", "text": "unknown op: " + op})
	}
}

func handleChat(c *wsConn, msg map[string]any) {
	name, _ := msg["model"].(string)
	text, _ := msg["text"].(string)
	n := 40
	if f, ok := msg["n"].(float64); ok && f > 0 {
		n = int(f)
	}
	if text == "" {
		sendJSON(c, map[string]any{"op": "error", "text": "empty prompt"})
		return
	}
	m := findModel(cfg.root, name)
	if m == nil {
		sendJSON(c, map[string]any{"op": "error", "text": "unknown model: " + name})
		return
	}
	reply, status, err := runChat(*m, text, n)
	if status != "" {
		sendJSON(c, map[string]any{"op": "status", "text": status})
	}
	if err != nil {
		sendJSON(c, map[string]any{"op": "error", "text": err.Error()})
		return
	}
	sendJSON(c, map[string]any{"op": "chat_result", "text": reply, "model": name})
}

func sendJSON(c *wsConn, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		return
	}
	c.writeText(b)
}
