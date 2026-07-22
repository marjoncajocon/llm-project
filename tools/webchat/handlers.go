// handlers.go - Media op handlers (image/audio generation) over the WebSocket.
package main

import (
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

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
