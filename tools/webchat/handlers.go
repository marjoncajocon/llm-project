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

// handleChat is the unified ("mixed") chat: it auto-detects whether the message
// asks for text / image / audio / video (or the client can force a modality) and
// routes accordingly, rendering the result inline. Text is real; image/audio use
// the toy teach-then-reproduce path (needs an attached target); video and true
// zero-shot text->media aren't available (no trained media model) — those return
// an honest note. The media branch is the seam to plug a real model into later.
//
// Message fields: model, text (raw user message), context (history-built prompt),
// modality ("auto"|"text"|"image"|"audio"|"video"), media_b64 (optional attachment).
func handleChat(c *wsConn, msg map[string]any) {
	name, _ := msg["model"].(string)
	text, _ := msg["text"].(string)
	context, _ := msg["context"].(string)
	if context == "" {
		context = text
	}
	modality, _ := msg["modality"].(string)
	attach, _ := msg["media_b64"].(string)
	n := intField(msg, "n", 512)
	if strings.TrimSpace(text) == "" {
		sendJSON(c, errMsg(fmt.Errorf("empty prompt")))
		return
	}
	if modality == "" || modality == "auto" {
		modality = detectMediaIntent(text)
	}

	switch modality {
	case "image", "audio":
		if attach != "" {
			chatMedia(c, modality, text, attach) // teach-then-reproduce, inline
		} else if modality == "image" && cfg.imagegen != "" {
			chatDiffusers(c, text) // real diffusion model (HF diffusers) via the hook
		} else if modality == "image" && t2iTrained() {
			chatT2I(c, text) // from-scratch text->image model
		} else {
			chatSay(c, mediaHint(modality))
		}
	case "video":
		chatSay(c, "🎬 I detected a video request. This engine has no video generation "+
			"model or encoder, so I can't make one. Text and toy image/audio work; a real "+
			"text→video model would plug in here.")
	default: // text
		m := findModel(cfg.root, name)
		if m == nil {
			sendJSON(c, errMsg(fmt.Errorf("unknown model: %s", name)))
			return
		}
		streamText(c, m, context, n)
	}
	sendJSON(c, map[string]any{"op": "chat_done", "model": name})
}

// detectMediaIntent is a lightweight keyword classifier for the "auto" modality.
// It's heuristic — the UI also offers a manual modality override.
func detectMediaIntent(text string) string {
	t := " " + strings.ToLower(text) + " "
	has := func(words ...string) bool {
		for _, w := range words {
			if strings.Contains(t, " "+w) || strings.Contains(t, w+" ") || strings.Contains(t, w+"s ") {
				return true
			}
		}
		return false
	}
	switch {
	case has("video", "movie", "clip", "animation", "gif"):
		return "video"
	case has("image", "picture", "photo", "draw", "drawing", "paint", "logo", "icon", "sketch", "selfie"):
		return "image"
	case has("audio", "sound", "music", "song", "beep", "tune", "voice", "melody", "noise"):
		return "audio"
	default:
		return "text"
	}
}

func chatSay(c *wsConn, s string) { sendJSON(c, map[string]any{"op": "chat_delta", "text": s}) }

func mediaHint(kind string) string {
	if kind == "image" {
		return "🎨 I detected an image request, but no trained text→image model is loaded. " +
			"Train one from scratch with `transformer t2i-train <captioned-image-dir> model/t2i/t2i` " +
			"(the built-in VQ-VAE + GPT text→image), then I'll generate here. Or attach a target " +
			"image with 📎 to run the teach-then-reproduce demo now."
	}
	return fmt.Sprintf("🎨 I detected an %s request. This engine has no trained text→%s model, "+
		"so I can't generate one from words alone. Attach a target %s with 📎 and I'll run the "+
		"teach-then-reproduce demo.", kind, kind, kind)
}

// chatDiffusers calls the configured external text->image command (e.g. the
// diffusers/SD-Turbo script) — the prompt on stdin, an output PNG path appended —
// and renders the returned image inline. This runs a real diffusion model beside
// the engine (the engine can't load diffusion checkpoints itself).
func chatDiffusers(c *wsConn, prompt string) {
	parts := strings.Fields(cfg.imagegen)
	if len(parts) == 0 {
		chatSay(c, "⚠ imagegen command is empty")
		return
	}
	dir, err := os.MkdirTemp("", "webchat")
	if err != nil {
		chatSay(c, "⚠ "+err.Error())
		return
	}
	defer os.RemoveAll(dir)
	out := filepath.Join(dir, "gen.png")
	chatSay(c, "🖼️ generating with the diffusion model… (first run downloads the model; CPU is slow)")
	args := append(append([]string{}, parts[1:]...), out)
	if _, se, err := runCmd(parts[0], args, prompt); err != nil {
		chatSay(c, "\n⚠ image generation failed: "+lastLine(strings.TrimSpace(se)))
		return
	}
	png, err := os.ReadFile(out)
	if err != nil {
		chatSay(c, "\n⚠ "+err.Error())
		return
	}
	sendJSON(c, map[string]any{"op": "chat_media", "kind": "image",
		"png_b64": base64.StdEncoding.EncodeToString(png)})
}

// t2iTrained reports whether a saved text->image model exists at model/t2i/.
func t2iTrained() bool {
	_, err := os.Stat(filepath.Join(cfg.root, "model", "t2i", "t2i.vqt"))
	return err == nil
}

// chatT2I generates an image from a caption using the pre-trained t2i model
// (fast: it loads the saved VQ-VAE + GPT, no retraining) and renders it inline.
func chatT2I(c *wsConn, prompt string) {
	dir, err := os.MkdirTemp("", "webchat")
	if err != nil {
		chatSay(c, "⚠ "+err.Error())
		return
	}
	defer os.RemoveAll(dir)
	out := filepath.Join(dir, "gen.ppm")
	if _, _, err := runCmd(cfg.exe, []string{"t2i-gen", "model/t2i/t2i", out, prompt}, ""); err != nil {
		chatSay(c, "⚠ "+err.Error())
		return
	}
	png, err := ppmFileToPNG(out)
	if err != nil {
		chatSay(c, "⚠ "+err.Error())
		return
	}
	sendJSON(c, map[string]any{"op": "chat_media", "kind": "image",
		"png_b64": base64.StdEncoding.EncodeToString(png)})
}

// chatMedia runs the toy text->media path on an attached target and renders the
// result inline in the chat. (This is where a real text->media model would slot.)
func chatMedia(c *wsConn, kind, prompt, attachB64 string) {
	data, err := decodeMediaB64(attachB64)
	if err != nil {
		chatSay(c, "⚠ "+err.Error())
		return
	}
	dir, err := os.MkdirTemp("", "webchat")
	if err != nil {
		chatSay(c, "⚠ "+err.Error())
		return
	}
	defer os.RemoveAll(dir)

	if kind == "image" {
		in, out := filepath.Join(dir, "t.ppm"), filepath.Join(dir, "o.ppm")
		if _, _, err := imageBytesToPPM(data, in, 16); err != nil {
			chatSay(c, "⚠ "+err.Error())
			return
		}
		if _, _, err := runCmd(cfg.exe, []string{"text2media", in, out, prompt, "250"}, ""); err != nil {
			chatSay(c, "⚠ "+err.Error())
			return
		}
		png, err := ppmFileToPNG(out)
		if err != nil {
			chatSay(c, "⚠ "+err.Error())
			return
		}
		sendJSON(c, map[string]any{"op": "chat_media", "kind": "image",
			"png_b64": base64.StdEncoding.EncodeToString(png)})
		return
	}
	// audio
	wav, _, err := prepWav(data, 256)
	if err != nil {
		chatSay(c, "⚠ "+err.Error())
		return
	}
	in, out := filepath.Join(dir, "t.wav"), filepath.Join(dir, "o.wav")
	if err := os.WriteFile(in, wav, 0o644); err != nil {
		chatSay(c, "⚠ "+err.Error())
		return
	}
	if _, _, err := runCmd(cfg.exe, []string{"text2media", in, out, prompt, "250"}, ""); err != nil {
		chatSay(c, "⚠ "+err.Error())
		return
	}
	outWav, err := os.ReadFile(out)
	if err != nil {
		chatSay(c, "⚠ "+err.Error())
		return
	}
	sendJSON(c, map[string]any{"op": "chat_media", "kind": "audio",
		"wav_b64": base64.StdEncoding.EncodeToString(outWav)})
}

// streamText routes the text path. Pure-C models: one delta. For gpt2/llama it
// prefers the engine's OWN tokenizer (hfgen — no Python) when the model's vocab
// is exported to model/<name>/tok.hf, and falls back to the Python bridges.
func streamText(c *wsConn, m *Model, prompt string, n int) {
	if m.Mode == "purec-text" {
		reply, _, err := runChat(*m, prompt, n)
		if err != nil {
			sendJSON(c, errMsg(err))
			return
		}
		if reply = trimAtStop(reply); reply != "" {
			sendJSON(c, map[string]any{"op": "chat_delta", "text": reply})
		}
		return
	}

	// Pure-C tokenizer path (no Python) when the vocab was exported.
	if _, err := os.Stat(filepath.Join(cfg.root, "model", m.Name, "tok.hf")); err == nil {
		rel := "model/" + m.Name + "/tok.hf"
		if err := runCmdStream(cfg.exe, []string{"hfgen", m.Ckpt, rel, strconv.Itoa(n)}, prompt,
			func(line string) { forwardDelta(c, line) }); err != nil {
			sendJSON(c, errMsg(err))
		}
		return
	}

	// Fallback: the Python tokenizer bridges.
	switch m.Mode {
	case "gpt2":
		if err := runCmdStream(cfg.python,
			[]string{"tools/gpt2_stream.py", "--name", m.Name}, prompt,
			func(line string) { forwardDelta(c, line) }); err != nil {
			sendJSON(c, errMsg(err))
		}
	case "llama":
		args := []string{"tools/llama_stream.py", "--ckpt", m.Ckpt, "--n", strconv.Itoa(n)}
		if m.HFRepo != "" {
			args = append(args, "--model", m.HFRepo)
		}
		if err := runCmdStream(cfg.python, args, prompt,
			func(line string) { forwardDelta(c, line) }); err != nil {
			sendJSON(c, errMsg(err))
		}
	default:
		sendJSON(c, errMsg(fmt.Errorf("unsupported model mode %q", m.Mode)))
	}
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
	for _, st := range []string{"<|endoftext|>", "<|im_end|>", "<|eot_id|>", "\nUser:", "\nHuman:", "\nYou:", "\nAssistant:", "\nAI:"} {
		if i := strings.Index(s, st); i >= 0 {
			s = s[:i]
		}
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
