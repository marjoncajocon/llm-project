// handlers.go - Media op handlers (image/audio generation) over the WebSocket.
package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// handleChat streams a reply for the client-built conversation prompt (text).
// Routes by model type: pure-C text models have their own tokenizer and are fast
// (one delta, no streaming); gpt2 / llama stream token deltas live via the Python
// wrappers (tiktoken / tokenizers do the incremental detokenization).
func handleChat(c *wsConn, msg map[string]any) {
	name, _ := msg["model"].(string)
	text, _ := msg["text"].(string)
	// Generous safety cap: generation stops naturally on EOS / the stop sequence
	// (handled in the stream wrappers); this only bounds a runaway base model.
	n := intField(msg, "n", 512)
	if strings.TrimSpace(text) == "" {
		sendJSON(c, errMsg(fmt.Errorf("empty prompt")))
		return
	}
	m := findModel(cfg.root, name)
	if m == nil {
		sendJSON(c, errMsg(fmt.Errorf("unknown model: %s", name)))
		return
	}

	switch m.Mode {
	case "purec-text":
		reply, _, err := runChat(*m, text, n)
		if err != nil {
			sendJSON(c, errMsg(err))
			return
		}
		if reply = trimAtStop(reply); reply != "" {
			sendJSON(c, map[string]any{"op": "chat_delta", "text": reply})
		}
	case "gpt2":
		if err := runCmdStream(cfg.python,
			[]string{"tools/gpt2_stream.py", "--name", m.Name}, text,
			func(line string) { forwardDelta(c, line) }); err != nil {
			sendJSON(c, errMsg(err))
			return
		}
	case "llama":
		args := []string{"tools/llama_stream.py", "--ckpt", m.Ckpt, "--n", strconv.Itoa(n)}
		if m.HFRepo != "" {
			args = append(args, "--model", m.HFRepo)
		}
		if err := runCmdStream(cfg.python, args, text,
			func(line string) { forwardDelta(c, line) }); err != nil {
			sendJSON(c, errMsg(err))
			return
		}
	default:
		sendJSON(c, errMsg(fmt.Errorf("unsupported model mode %q", m.Mode)))
		return
	}
	sendJSON(c, map[string]any{"op": "chat_done", "model": name})
}

// forwardDelta parses a wrapper JSON line ({"d":"piece"}) and forwards the text.
func forwardDelta(c *wsConn, line string) {
	line = strings.TrimSpace(line)
	if line == "" {
		return
	}
	var m map[string]any
	if json.Unmarshal([]byte(line), &m) != nil {
		return
	}
	if d, ok := m["d"].(string); ok && d != "" {
		sendJSON(c, map[string]any{"op": "chat_delta", "text": d})
	}
}

// trimAtStop cuts a reply at the first hallucinated next turn (the pure-C path
// generates a fixed token count with no stop sequence).
func trimAtStop(s string) string {
	if i := strings.Index(s, "\nUser"); i >= 0 {
		s = s[:i]
	}
	return strings.TrimRight(s, "\n ")
}

// imggen: browser image -> P6 PPM (resized) -> engine reconstruct -> PNG back.
func handleImggen(c *wsConn, msg map[string]any) {
	data, err := decodeMediaB64(msg["png_b64"])
	if err != nil {
		sendJSON(c, errMsg(err))
		return
	}
	steps := intField(msg, "steps", 100)
	levels := intField(msg, "levels", 4)
	dir, err := os.MkdirTemp("", "webchat")
	if err != nil {
		sendJSON(c, errMsg(err))
		return
	}
	defer os.RemoveAll(dir)

	in := filepath.Join(dir, "in.ppm")
	out := filepath.Join(dir, "out.ppm")
	// Cap at 16x16: attention is O(tokens^2) and tokens == pixels, so larger
	// inputs get very slow on CPU. 256 tokens keeps a reconstruct in ~15-20s.
	w, h, err := imageBytesToPPM(data, in, 16)
	if err != nil {
		sendJSON(c, errMsg(err))
		return
	}
	_, se, err := runCmd(cfg.exe, []string{"imggen", in, out, strconv.Itoa(levels), strconv.Itoa(steps)}, "")
	if se != "" {
		sendJSON(c, statusMsg(se))
	}
	if err != nil {
		sendJSON(c, errMsg(err))
		return
	}
	inPNG, _ := ppmFileToPNG(in)
	outPNG, err := ppmFileToPNG(out)
	if err != nil {
		sendJSON(c, errMsg(err))
		return
	}
	sendJSON(c, map[string]any{
		"op":        "image_result",
		"input_png": base64.StdEncoding.EncodeToString(inPNG),
		"png_b64":   base64.StdEncoding.EncodeToString(outPNG),
		"w":         w, "h": h,
	})
}

// audgen: browser WAV -> 16-bit mono, trimmed -> engine reconstruct -> WAV back.
func handleAudgen(c *wsConn, msg map[string]any) {
	data, err := decodeMediaB64(msg["wav_b64"])
	if err != nil {
		sendJSON(c, errMsg(err))
		return
	}
	steps := intField(msg, "steps", 100)
	// Cap at 256 samples (~16 ms @16 kHz): attention is O(samples^2) on CPU.
	wav, frames, err := prepWav(data, 256)
	if err != nil {
		sendJSON(c, errMsg(err))
		return
	}
	dir, err := os.MkdirTemp("", "webchat")
	if err != nil {
		sendJSON(c, errMsg(err))
		return
	}
	defer os.RemoveAll(dir)

	in := filepath.Join(dir, "in.wav")
	out := filepath.Join(dir, "out.wav")
	if err := os.WriteFile(in, wav, 0o644); err != nil {
		sendJSON(c, errMsg(err))
		return
	}
	_, se, err := runCmd(cfg.exe, []string{"audgen", in, out, strconv.Itoa(steps)}, "")
	if se != "" {
		sendJSON(c, statusMsg(se))
	}
	if err != nil {
		sendJSON(c, errMsg(err))
		return
	}
	outWav, err := os.ReadFile(out)
	if err != nil {
		sendJSON(c, errMsg(err))
		return
	}
	sendJSON(c, map[string]any{
		"op":      "audio_result",
		"wav_b64": base64.StdEncoding.EncodeToString(outWav),
		"frames":  frames,
	})
}

// caption: teach-then-reproduce media->text. Builds the media file, runs the C
// `caption` subcommand (train a tiny multimodal model to caption it, then
// regenerate the caption), and returns the generated text. Video also returns a
// frames->GIF preview of what was ingested.
func handleCaption(c *wsConn, msg map[string]any) {
	kind, _ := msg["kind"].(string)
	captionText, _ := msg["caption"].(string)
	if strings.TrimSpace(captionText) == "" {
		sendJSON(c, errMsg(fmt.Errorf("enter a caption to teach the model")))
		return
	}
	steps := intField(msg, "steps", 250)
	data, err := decodeMediaB64(msg["media_b64"])
	if err != nil {
		sendJSON(c, errMsg(err))
		return
	}
	dir, err := os.MkdirTemp("", "webchat")
	if err != nil {
		sendJSON(c, errMsg(err))
		return
	}
	defer os.RemoveAll(dir)

	var mediaPath, gifB64 string
	switch kind {
	case "image":
		mediaPath = filepath.Join(dir, "m.ppm")
		if _, _, err := imageBytesToPPM(data, mediaPath, 16); err != nil {
			sendJSON(c, errMsg(err))
			return
		}
	case "audio":
		wav, _, err := prepWav(data, 256)
		if err != nil {
			sendJSON(c, errMsg(err))
			return
		}
		mediaPath = filepath.Join(dir, "m.wav")
		if err := os.WriteFile(mediaPath, wav, 0o644); err != nil {
			sendJSON(c, errMsg(err))
			return
		}
	case "video":
		vid := filepath.Join(dir, "in.mp4")
		if err := os.WriteFile(vid, data, 0o644); err != nil {
			sendJSON(c, errMsg(err))
			return
		}
		framesDir := filepath.Join(dir, "frames")
		os.MkdirAll(framesDir, 0o755)
		_, se, err := runCmd(cfg.python, []string{"tools/video_to_frames.py", vid, framesDir, "--fps", "2", "--size", "16"}, "")
		if err != nil {
			sendJSON(c, errMsg(fmt.Errorf("frame extraction failed (video needs ffmpeg on PATH): %s", firstNonEmptyLine(se))))
			return
		}
		mediaPath = framesDir
		if g, err := framesToGIF(framesDir, 32); err == nil {
			gifB64 = base64.StdEncoding.EncodeToString(g)
		}
	default:
		sendJSON(c, errMsg(fmt.Errorf("unknown caption kind %q", kind)))
		return
	}

	out, se, err := runCmd(cfg.exe, []string{"caption", mediaPath, captionText, strconv.Itoa(steps)}, "")
	if se != "" {
		sendJSON(c, statusMsg(se))
	}
	if err != nil {
		sendJSON(c, errMsg(err))
		return
	}
	text := ""
	for _, ln := range strings.Split(out, "\n") {
		ln = strings.TrimRight(ln, "\r")
		if strings.HasPrefix(ln, "caption: ") {
			text = strings.TrimPrefix(ln, "caption: ")
			break
		}
	}
	res := map[string]any{"op": "caption_result", "text": text, "kind": kind}
	if gifB64 != "" {
		res["gif_b64"] = gifB64
	}
	sendJSON(c, res)
}

// text2media: teach-then-reproduce text -> image / audio. Uses the uploaded
// media as the target the model learns to generate from the text prompt, then
// returns the regenerated media (which reproduces the target).
func handleText2media(c *wsConn, msg map[string]any) {
	kind, _ := msg["kind"].(string)
	text, _ := msg["text"].(string)
	if strings.TrimSpace(text) == "" {
		sendJSON(c, errMsg(fmt.Errorf("enter a text prompt")))
		return
	}
	steps := intField(msg, "steps", 300)
	data, err := decodeMediaB64(msg["media_b64"])
	if err != nil {
		sendJSON(c, errMsg(err))
		return
	}
	dir, err := os.MkdirTemp("", "webchat")
	if err != nil {
		sendJSON(c, errMsg(err))
		return
	}
	defer os.RemoveAll(dir)

	switch kind {
	case "image":
		in := filepath.Join(dir, "t.ppm")
		out := filepath.Join(dir, "o.ppm")
		w, h, err := imageBytesToPPM(data, in, 16)
		if err != nil {
			sendJSON(c, errMsg(err))
			return
		}
		_, se, err := runCmd(cfg.exe, []string{"text2media", in, out, text, strconv.Itoa(steps)}, "")
		if se != "" {
			sendJSON(c, statusMsg(se))
		}
		if err != nil {
			sendJSON(c, errMsg(err))
			return
		}
		inPNG, _ := ppmFileToPNG(in)
		outPNG, err := ppmFileToPNG(out)
		if err != nil {
			sendJSON(c, errMsg(err))
			return
		}
		sendJSON(c, map[string]any{"op": "image_result",
			"input_png": base64.StdEncoding.EncodeToString(inPNG),
			"png_b64":   base64.StdEncoding.EncodeToString(outPNG), "w": w, "h": h})
	case "audio":
		wav, _, err := prepWav(data, 256)
		if err != nil {
			sendJSON(c, errMsg(err))
			return
		}
		in := filepath.Join(dir, "t.wav")
		out := filepath.Join(dir, "o.wav")
		if err := os.WriteFile(in, wav, 0o644); err != nil {
			sendJSON(c, errMsg(err))
			return
		}
		_, se, err := runCmd(cfg.exe, []string{"text2media", in, out, text, strconv.Itoa(steps)}, "")
		if se != "" {
			sendJSON(c, statusMsg(se))
		}
		if err != nil {
			sendJSON(c, errMsg(err))
			return
		}
		outWav, err := os.ReadFile(out)
		if err != nil {
			sendJSON(c, errMsg(err))
			return
		}
		sendJSON(c, map[string]any{"op": "audio_result",
			"wav_b64": base64.StdEncoding.EncodeToString(outWav)})
	default:
		sendJSON(c, errMsg(fmt.Errorf("text2media kind must be image or audio")))
	}
}

func decodeMediaB64(v any) ([]byte, error) {
	s, _ := v.(string)
	if s == "" {
		return nil, fmt.Errorf("no media payload")
	}
	if i := strings.Index(s, "base64,"); i >= 0 { // strip a data-URL prefix
		s = s[i+7:]
	}
	return base64.StdEncoding.DecodeString(s)
}

func intField(m map[string]any, k string, def int) int {
	if f, ok := m[k].(float64); ok && f > 0 {
		return int(f)
	}
	return def
}

func errMsg(err error) map[string]any    { return map[string]any{"op": "error", "text": err.Error()} }
func statusMsg(s string) map[string]any  { return map[string]any{"op": "status", "text": s} }
