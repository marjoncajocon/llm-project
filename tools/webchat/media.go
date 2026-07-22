// media.go - Format bridging between browser-friendly media (PNG/JPEG, WAV) and
// what the C engine reads/writes (binary P6 PPM, 16-bit PCM WAV). Go stdlib only.
package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color/palette"
	"image/draw"
	"image/gif"
	_ "image/jpeg" // register decoder for image.Decode
	"image/png"
	"io"
	"os"
	"path/filepath"
)

// ---- images -------------------------------------------------------------

// imageBytesToPPM decodes PNG/JPEG/GIF bytes, resizes to fit maxDim (keeping
// aspect, so w*h stays within the engine's pixel cap), and writes a P6 PPM.
func imageBytesToPPM(data []byte, path string, maxDim int) (int, int, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return 0, 0, fmt.Errorf("decode image: %w", err)
	}
	small, w, h := resizeFit(img, maxDim)
	return w, h, writePPMFile(path, small, w, h)
}

func resizeFit(src image.Image, maxDim int) (image.Image, int, int) {
	b := src.Bounds()
	sw, sh := b.Dx(), b.Dy()
	if sw == 0 || sh == 0 {
		return src, sw, sh
	}
	dw, dh := sw, sh
	if sw > maxDim || sh > maxDim {
		if sw >= sh {
			dw, dh = maxDim, sh*maxDim/sw
		} else {
			dw, dh = sw*maxDim/sh, maxDim
		}
	}
	if dw < 1 {
		dw = 1
	}
	if dh < 1 {
		dh = 1
	}
	dst := image.NewRGBA(image.Rect(0, 0, dw, dh))
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			sx := b.Min.X + x*sw/dw
			sy := b.Min.Y + y*sh/dh
			dst.Set(x, y, src.At(sx, sy))
		}
	}
	return dst, dw, dh
}

func writePPMFile(path string, img image.Image, w, h int) error {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "P6\n%d %d\n255\n", w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x, y).RGBA() // 16-bit channels
			buf.WriteByte(byte(r >> 8))
			buf.WriteByte(byte(g >> 8))
			buf.WriteByte(byte(b >> 8))
		}
	}
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

// ppmFileToPNG reads a binary P6 PPM and re-encodes it as PNG.
func ppmFileToPNG(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	img, err := decodeP6(data)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decodeP6(data []byte) (image.Image, error) {
	r := bufio.NewReader(bytes.NewReader(data))
	magic, err := ppmToken(r)
	if err != nil || magic != "P6" {
		return nil, errors.New("not a binary P6 PPM")
	}
	w, err1 := ppmInt(r)
	h, err2 := ppmInt(r)
	mv, err3 := ppmInt(r)
	if err1 != nil || err2 != nil || err3 != nil {
		return nil, errors.New("bad PPM header")
	}
	if mv != 255 || w <= 0 || h <= 0 {
		return nil, fmt.Errorf("unsupported PPM (maxval=%d, %dx%d)", mv, w, h)
	}
	pix := make([]byte, w*h*3)
	if _, err := io.ReadFull(r, pix); err != nil {
		return nil, fmt.Errorf("short PPM pixel data: %w", err)
	}
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < w*h; i++ {
		img.Pix[i*4+0] = pix[i*3+0]
		img.Pix[i*4+1] = pix[i*3+1]
		img.Pix[i*4+2] = pix[i*3+2]
		img.Pix[i*4+3] = 255
	}
	return img, nil
}

// ppmToken reads one whitespace/comment-delimited token, consuming the single
// trailing delimiter (so after the maxval token the reader sits on pixel data).
func ppmToken(r *bufio.Reader) (string, error) {
	for {
		b, err := r.ReadByte()
		if err != nil {
			return "", err
		}
		if b == '#' {
			for {
				c, err := r.ReadByte()
				if err != nil {
					return "", err
				}
				if c == '\n' {
					break
				}
			}
			continue
		}
		if b == ' ' || b == '\t' || b == '\n' || b == '\r' {
			continue
		}
		tok := []byte{b}
		for {
			c, err := r.ReadByte()
			if err != nil {
				break
			}
			if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
				break
			}
			tok = append(tok, c)
		}
		return string(tok), nil
	}
}

func ppmInt(r *bufio.Reader) (int, error) {
	tok, err := ppmToken(r)
	if err != nil {
		return 0, err
	}
	n := 0
	for _, c := range tok {
		if c < '0' || c > '9' {
			return 0, fmt.Errorf("non-numeric PPM token %q", tok)
		}
		n = n*10 + int(c-'0')
	}
	return n, nil
}

// framesToGIF assembles frame_0000.ppm, frame_0001.ppm, ... (up to maxFrames)
// from dir into an animated GIF for a preview of what the engine ingested.
func framesToGIF(dir string, maxFrames int) ([]byte, error) {
	g := &gif.GIF{}
	for i := 0; i < maxFrames; i++ {
		data, err := os.ReadFile(filepath.Join(dir, fmt.Sprintf("frame_%04d.ppm", i)))
		if err != nil {
			break
		}
		img, err := decodeP6(data)
		if err != nil {
			break
		}
		pal := image.NewPaletted(img.Bounds(), palette.Plan9)
		draw.Draw(pal, pal.Bounds(), img, img.Bounds().Min, draw.Src)
		g.Image = append(g.Image, pal)
		g.Delay = append(g.Delay, 40) // 0.4s per frame
	}
	if len(g.Image) == 0 {
		return nil, errors.New("no frames to render")
	}
	var buf bytes.Buffer
	if err := gif.EncodeAll(&buf, g); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// ---- audio --------------------------------------------------------------

// prepWav validates a 16-bit PCM WAV, downmixes to mono, trims to maxSamples,
// and re-emits a clean canonical WAV that the engine's audgen accepts.
func prepWav(data []byte, maxSamples int) ([]byte, int, error) {
	af, ch, rate, bits, pcm, err := parseWav(data)
	if err != nil {
		return nil, 0, err
	}
	if af != 1 || bits != 16 {
		return nil, 0, fmt.Errorf("need a 16-bit PCM WAV (got format %d, %d-bit)", af, bits)
	}
	if ch < 1 {
		ch = 1
	}
	if rate <= 0 {
		rate = 16000
	}
	frames := len(pcm) / (2 * ch)
	if frames > maxSamples {
		frames = maxSamples
	}
	mono := make([]int16, frames)
	for i := 0; i < frames; i++ {
		sum := 0
		for c := 0; c < ch; c++ {
			off := (i*ch + c) * 2
			sum += int(int16(binary.LittleEndian.Uint16(pcm[off:])))
		}
		mono[i] = int16(sum / ch)
	}
	return buildWav(mono, rate), frames, nil
}

func parseWav(data []byte) (audioFmt, channels, rate, bits int, pcm []byte, err error) {
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		err = errors.New("not a RIFF/WAVE file")
		return
	}
	pos := 12
	for pos+8 <= len(data) {
		id := string(data[pos : pos+4])
		sz := int(binary.LittleEndian.Uint32(data[pos+4 : pos+8]))
		pos += 8
		if sz < 0 || pos+sz > len(data) {
			sz = len(data) - pos
		}
		switch id {
		case "fmt ":
			if sz >= 16 {
				audioFmt = int(binary.LittleEndian.Uint16(data[pos:]))
				channels = int(binary.LittleEndian.Uint16(data[pos+2:]))
				rate = int(binary.LittleEndian.Uint32(data[pos+4:]))
				bits = int(binary.LittleEndian.Uint16(data[pos+14:]))
			}
		case "data":
			pcm = data[pos : pos+sz]
		}
		pos += sz + (sz & 1) // chunks are word-aligned
	}
	return
}

func buildWav(samples []int16, rate int) []byte {
	dataSize := len(samples) * 2
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+dataSize))
	b.WriteString("WAVE")
	b.WriteString("fmt ")
	binary.Write(&b, binary.LittleEndian, uint32(16))
	binary.Write(&b, binary.LittleEndian, uint16(1)) // PCM
	binary.Write(&b, binary.LittleEndian, uint16(1)) // mono
	binary.Write(&b, binary.LittleEndian, uint32(rate))
	binary.Write(&b, binary.LittleEndian, uint32(rate*2)) // byte rate
	binary.Write(&b, binary.LittleEndian, uint16(2))      // block align
	binary.Write(&b, binary.LittleEndian, uint16(16))     // bits
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(dataSize))
	for _, s := range samples {
		binary.Write(&b, binary.LittleEndian, s)
	}
	return b.Bytes()
}
