// doc.go - a thin typesetting layer over gofpdf: headings, paragraphs, code
// blocks, tables, callouts, and a table of contents with real page numbers.
//
// Everything written through here stays ASCII on purpose. gofpdf's built-in
// fonts are cp1252, so an em dash or a curly quote pasted from a terminal turns
// into mojibake in the output rather than failing loudly.
package main

import (
	"strings"

	"github.com/jung-kurt/gofpdf"
)

const (
	pageW   = 210.0 // A4 portrait, mm
	margL   = 18.0
	margR   = 18.0
	margT   = 18.0
	margB   = 20.0
	bodyW   = pageW - margL - margR
	tocPage = 2 // pages RESERVED for the table of contents, fixed across both passes
)

type entry struct {
	lvl   int
	title string
	page  int
}

type doc struct {
	pdf *gofpdf.Fpdf
	// pass 1 records headings here; pass 2 reads them back to print the TOC
	rec    *[]entry
	toc    []entry
	secNum [3]int
}

func newDoc(toc []entry, rec *[]entry) *doc {
	pdf := gofpdf.New("P", "mm", "A4", "")
	pdf.SetMargins(margL, margT, margR)
	pdf.SetAutoPageBreak(true, margB)
	d := &doc{pdf: pdf, rec: rec, toc: toc}
	pdf.SetFooterFunc(func() {
		if pdf.PageNo() == 1 {
			return // no chrome on the title page
		}
		pdf.SetY(-14)
		pdf.SetFont("Helvetica", "", 8)
		pdf.SetTextColor(120, 120, 120)
		pdf.CellFormat(bodyW/2, 6, "Pure-C Transformer Engine - Complete Guide", "", 0, "L", false, 0, "")
		pdf.CellFormat(bodyW/2, 6, itoa(pdf.PageNo()), "", 0, "R", false, 0, "")
		pdf.SetTextColor(0, 0, 0)
	})
	return d
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// ---------------------------------------------------------------------------
// Headings
// ---------------------------------------------------------------------------

// heading records itself on pass 1 and renders on both passes, so the page
// numbers gathered in pass 1 are the ones the pass-2 TOC prints.
func (d *doc) heading(lvl int, title string) {
	switch lvl {
	case 1:
		d.secNum[0]++
		d.secNum[1], d.secNum[2] = 0, 0
	case 2:
		d.secNum[1]++
		d.secNum[2] = 0
	case 3:
		d.secNum[2]++
	}
	num := itoa(d.secNum[0])
	if lvl >= 2 {
		num += "." + itoa(d.secNum[1])
	}
	if lvl >= 3 {
		num += "." + itoa(d.secNum[2])
	}
	full := num + "  " + title

	if lvl == 1 {
		d.pdf.AddPage()
	} else {
		// Keep a heading with the text that follows it: if there is not enough
		// room for the heading plus a couple of lines, start a new page.
		if d.pdf.GetY() > 297-margB-28 {
			d.pdf.AddPage()
		}
		d.pdf.Ln(3)
	}

	if d.rec != nil {
		*d.rec = append(*d.rec, entry{lvl: lvl, title: full, page: d.pdf.PageNo()})
	}

	switch lvl {
	case 1:
		d.pdf.SetFillColor(28, 42, 64)
		d.pdf.SetTextColor(255, 255, 255)
		d.pdf.SetFont("Helvetica", "B", 16)
		d.pdf.CellFormat(bodyW, 11, "  "+full, "", 1, "L", true, 0, "")
		d.pdf.SetTextColor(0, 0, 0)
		d.pdf.Ln(4)
	case 2:
		d.pdf.SetFont("Helvetica", "B", 12.5)
		d.pdf.SetTextColor(28, 42, 64)
		d.pdf.CellFormat(bodyW, 7, full, "", 1, "L", false, 0, "")
		d.pdf.SetDrawColor(200, 208, 220)
		y := d.pdf.GetY()
		d.pdf.Line(margL, y, margL+bodyW, y)
		d.pdf.SetTextColor(0, 0, 0)
		d.pdf.Ln(2.5)
	default:
		d.pdf.SetFont("Helvetica", "B", 10.5)
		d.pdf.CellFormat(bodyW, 6, full, "", 1, "L", false, 0, "")
		d.pdf.Ln(0.5)
	}
}

func (d *doc) h1(t string) { d.heading(1, t) }
func (d *doc) h2(t string) { d.heading(2, t) }
func (d *doc) h3(t string) { d.heading(3, t) }

// ---------------------------------------------------------------------------
// Body text
// ---------------------------------------------------------------------------

// p renders a paragraph. A leading "*" on a word pair marks bold inline runs is
// deliberately NOT supported - keeping one text style per call keeps the layout
// predictable; use lead() for emphasis instead.
func (d *doc) p(text string) {
	d.pdf.SetFont("Helvetica", "", 10)
	d.pdf.SetTextColor(20, 20, 20)
	d.pdf.MultiCell(bodyW, 5.0, text, "", "L", false)
	d.pdf.Ln(1.6)
}

// lead is a paragraph in bold - for the one sentence that matters most.
func (d *doc) lead(text string) {
	d.pdf.SetFont("Helvetica", "B", 10)
	d.pdf.SetTextColor(20, 20, 20)
	d.pdf.MultiCell(bodyW, 5.0, text, "", "L", false)
	d.pdf.Ln(1.6)
}

func (d *doc) bullet(text string) {
	d.pdf.SetFont("Helvetica", "", 10)
	x := d.pdf.GetX()
	d.pdf.SetX(x + 3)
	d.pdf.CellFormat(3.5, 5, "-", "", 0, "L", false, 0, "")
	d.pdf.MultiCell(bodyW-7, 5, text, "", "L", false)
	d.pdf.SetX(x)
	d.pdf.Ln(0.6)
}

// step is a numbered instruction, for the walkthroughs.
func (d *doc) step(n int, text string) {
	d.pdf.SetFont("Helvetica", "B", 10)
	x := d.pdf.GetX()
	d.pdf.SetX(x + 1)
	d.pdf.CellFormat(7, 5, itoa(n)+".", "", 0, "L", false, 0, "")
	d.pdf.SetFont("Helvetica", "", 10)
	d.pdf.MultiCell(bodyW-8, 5, text, "", "L", false)
	d.pdf.SetX(x)
	d.pdf.Ln(0.8)
}

// ---------------------------------------------------------------------------
// Code blocks
// ---------------------------------------------------------------------------

func (d *doc) code(lines ...string) {
	const lh = 4.3
	d.pdf.SetFont("Courier", "", 8.3)
	// Wrap first so the background box is the right height.
	var out []string
	for _, ln := range lines {
		if ln == "" {
			out = append(out, "")
			continue
		}
		for _, w := range d.pdf.SplitLines([]byte(ln), bodyW-6) {
			out = append(out, string(w))
		}
	}
	h := float64(len(out))*lh + 3.4
	// Any block that COULD fit on a page of its own gets one, rather than being
	// split. Blocks too tall for that are allowed to flow across the break.
	avail := 297 - margB - d.pdf.GetY()
	full := 297 - margT - margB - 6
	if h > avail && h <= full {
		d.pdf.AddPage()
	}

	// The background is painted PER LINE rather than as one rectangle behind the
	// whole block. A single rect is drawn on the page it starts on, so a block
	// that flows onto the next page loses its background there entirely - which
	// is exactly what happened to the long C example.
	d.pdf.SetFillColor(244, 246, 249)
	d.pdf.SetTextColor(24, 32, 48)
	d.pdf.SetX(margL)
	d.pdf.CellFormat(bodyW, 1.7, "", "", 1, "L", true, 0, "")
	for _, ln := range out {
		d.pdf.SetX(margL)
		d.pdf.CellFormat(bodyW, lh, "  "+ln, "", 1, "L", true, 0, "")
	}
	d.pdf.SetX(margL)
	d.pdf.CellFormat(bodyW, 1.7, "", "", 1, "L", true, 0, "")
	d.pdf.SetTextColor(0, 0, 0)
	d.pdf.Ln(2.2)
}

// ---------------------------------------------------------------------------
// Callouts
// ---------------------------------------------------------------------------

type calloutKind int

const (
	noteBox calloutKind = iota
	warnBox
	tipBox
)

func (d *doc) callout(kind calloutKind, title, text string) {
	var r, g, b int
	var label string
	switch kind {
	case warnBox:
		r, g, b, label = 250, 240, 235, "WARNING: "
	case tipBox:
		r, g, b, label = 236, 247, 240, "TIP: "
	default:
		r, g, b, label = 238, 243, 250, "NOTE: "
	}
	body := label + title
	if text != "" {
		body += " " + text
	}
	d.pdf.SetFont("Helvetica", "", 9.4)
	lines := d.pdf.SplitLines([]byte(body), bodyW-10)
	h := float64(len(lines))*4.6 + 3.6
	if d.pdf.GetY()+h > 297-margB {
		d.pdf.AddPage()
	}
	y := d.pdf.GetY()
	d.pdf.SetFillColor(r, g, b)
	d.pdf.SetDrawColor(r-30, g-30, b-25)
	d.pdf.Rect(margL, y, bodyW, h, "F")
	// Accent bar down the left edge.
	switch kind {
	case warnBox:
		d.pdf.SetFillColor(200, 90, 40)
	case tipBox:
		d.pdf.SetFillColor(50, 150, 90)
	default:
		d.pdf.SetFillColor(60, 110, 190)
	}
	d.pdf.Rect(margL, y, 1.6, h, "F")
	d.pdf.SetXY(margL+5, y+1.8)
	d.pdf.SetTextColor(30, 30, 30)
	for _, ln := range lines {
		d.pdf.SetX(margL + 5)
		d.pdf.CellFormat(bodyW-10, 4.6, string(ln), "", 1, "L", false, 0, "")
	}
	d.pdf.SetTextColor(0, 0, 0)
	d.pdf.SetY(y + h)
	d.pdf.Ln(2.4)
}

func (d *doc) note(t string) { d.callout(noteBox, t, "") }
func (d *doc) warn(t string) { d.callout(warnBox, t, "") }
func (d *doc) tip(t string)  { d.callout(tipBox, t, "") }

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

// table draws a header row plus wrapped body rows. widths are in mm and should
// sum to bodyW. The first column is rendered monospace, since in this document
// it is always a flag, a field name or a file name.
func (d *doc) table(headers []string, widths []float64, rows [][]string) {
	const lh = 4.2

	drawHeader := func() {
		d.pdf.SetFont("Helvetica", "B", 8.6)
		d.pdf.SetFillColor(28, 42, 64)
		d.pdf.SetTextColor(255, 255, 255)
		for i, hd := range headers {
			d.pdf.CellFormat(widths[i], 6, " "+hd, "", 0, "L", true, 0, "")
		}
		d.pdf.Ln(6)
		d.pdf.SetTextColor(0, 0, 0)
	}

	if d.pdf.GetY() > 297-margB-24 {
		d.pdf.AddPage()
	}
	drawHeader()

	shade := false
	for _, row := range rows {
		// Measure the tallest cell in this row.
		maxLines := 1
		cellLines := make([][]string, len(row))
		for i, cell := range row {
			if i == 0 {
				d.pdf.SetFont("Courier", "", 8.0)
			} else {
				d.pdf.SetFont("Helvetica", "", 8.4)
			}
			var ls []string
			for _, w := range d.pdf.SplitLines([]byte(cell), widths[i]-2.4) {
				ls = append(ls, string(w))
			}
			if len(ls) == 0 {
				ls = []string{""}
			}
			cellLines[i] = ls
			if len(ls) > maxLines {
				maxLines = len(ls)
			}
		}
		h := float64(maxLines)*lh + 1.6

		if d.pdf.GetY()+h > 297-margB {
			d.pdf.AddPage()
			drawHeader()
		}

		y := d.pdf.GetY()
		if shade {
			d.pdf.SetFillColor(245, 247, 250)
			d.pdf.Rect(margL, y, bodyW, h, "F")
		}
		shade = !shade

		x := margL
		for i, ls := range cellLines {
			if i == 0 {
				d.pdf.SetFont("Courier", "", 8.0)
				d.pdf.SetTextColor(140, 40, 40)
			} else {
				d.pdf.SetFont("Helvetica", "", 8.4)
				d.pdf.SetTextColor(20, 20, 20)
			}
			for j, ln := range ls {
				d.pdf.SetXY(x+1.2, y+0.8+float64(j)*lh)
				d.pdf.CellFormat(widths[i]-2.4, lh, ln, "", 0, "L", false, 0, "")
			}
			x += widths[i]
		}
		d.pdf.SetTextColor(0, 0, 0)
		d.pdf.SetDrawColor(226, 230, 238)
		d.pdf.Line(margL, y+h, margL+bodyW, y+h)
		d.pdf.SetY(y + h)
	}
	d.pdf.Ln(3)
}

// ---------------------------------------------------------------------------
// Title page + table of contents
// ---------------------------------------------------------------------------

func (d *doc) titlePage(version, date string) {
	d.pdf.AddPage()
	d.pdf.SetFillColor(28, 42, 64)
	d.pdf.Rect(0, 0, pageW, 78, "F")

	d.pdf.SetY(24)
	d.pdf.SetTextColor(255, 255, 255)
	d.pdf.SetFont("Helvetica", "B", 26)
	d.pdf.CellFormat(pageW, 12, "Pure-C Transformer Engine", "", 1, "C", false, 0, "")
	d.pdf.SetFont("Helvetica", "", 15)
	d.pdf.CellFormat(pageW, 9, "The Complete Guide", "", 1, "C", false, 0, "")
	d.pdf.Ln(3)
	d.pdf.SetFont("Helvetica", "", 10)
	d.pdf.CellFormat(pageW, 6, "Build - Train - Fine-tune - Reinforce - Deploy - Embed", "", 1, "C", false, 0, "")

	d.pdf.SetTextColor(0, 0, 0)
	d.pdf.SetY(92)
	d.pdf.SetFont("Helvetica", "", 10.5)
	d.pdf.MultiCell(bodyW, 5.4,
		"A GPT-2 / LLaMA-family transformer written from scratch in portable C11: no BLAS, no "+
			"CUDA toolkit, no Python at runtime, and no third-party headers anywhere in the tree. "+
			"It trains, fine-tunes with LoRA, reinforces with GRPO, quantizes to int8, runs a "+
			"tool-using agent, and now optionally offloads its matrix multiplies to a GPU through "+
			"an OpenCL driver it loads at runtime.\n\n"+
			"This guide covers every step end to end: compiling each build flavour, preparing data, "+
			"pre-training, supervised fine-tuning, LoRA, GRPO reinforcement learning, GPU "+
			"acceleration, running on Google Colab, and linking the engine into your own C program. "+
			"Every command-line parameter is documented with its default, its effect, and when you "+
			"would actually reach for it.", "", "L", false)

	d.pdf.Ln(6)
	d.pdf.SetFont("Helvetica", "B", 10)
	d.pdf.CellFormat(bodyW, 6, "Read this first", "", 1, "L", false, 0, "")
	d.pdf.SetFont("Helvetica", "", 10)
	d.pdf.MultiCell(bodyW, 5.2,
		"Section 3 gets you a working binary. Section 6 trains a model. Section 10 links it into "+
			"your own program. If you only have ten minutes, read 3.1, 6.2 and 10.4.", "", "L", false)

	d.pdf.SetY(252)
	d.pdf.SetDrawColor(200, 208, 220)
	d.pdf.Line(margL, 252, margL+bodyW, 252)
	d.pdf.Ln(3)
	d.pdf.SetFont("Helvetica", "", 9)
	d.pdf.SetTextColor(90, 90, 90)
	d.pdf.MultiCell(bodyW, 4.8,
		"Version: "+version+"\nGenerated: "+date+
			"\nGenerated by tools/guidepdf (Go + gofpdf). Regenerate with: cd tools/guidepdf && go run .", "", "L", false)
	d.pdf.SetTextColor(0, 0, 0)
}

// renderTOC prints the contents pages, always consuming exactly tocPage pages so
// that the page numbers recorded in pass 1 are still valid in pass 2.
func (d *doc) renderTOC() {
	start := 0
	d.pdf.AddPage()
	start = d.pdf.PageNo()

	d.pdf.SetFont("Helvetica", "B", 16)
	d.pdf.SetTextColor(28, 42, 64)
	d.pdf.CellFormat(bodyW, 10, "Contents", "", 1, "L", false, 0, "")
	d.pdf.SetTextColor(0, 0, 0)
	d.pdf.Ln(2)

	for _, e := range d.toc {
		if e.lvl == 3 {
			continue // keep the contents to two levels so it fits
		}
		indent := 0.0
		if e.lvl == 2 {
			d.pdf.SetFont("Helvetica", "", 9)
			indent = 6
		} else {
			d.pdf.Ln(1.2)
			d.pdf.SetFont("Helvetica", "B", 10)
		}
		d.pdf.SetX(margL + indent)
		w := bodyW - indent - 12
		d.pdf.CellFormat(w, 5, e.title, "", 0, "L", false, 0, "")
		d.pdf.SetFont("Helvetica", "", 9)
		d.pdf.CellFormat(12, 5, itoa(e.page), "", 1, "R", false, 0, "")
	}

	// Pad out to the reserved page count.
	for d.pdf.PageNo()-start+1 < tocPage {
		d.pdf.AddPage()
	}
}

func wrapJoin(parts []string) string { return strings.Join(parts, " ") }
