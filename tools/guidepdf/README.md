# guidepdf - generates `build/guide.pdf`

The complete end-to-end manual for the engine, as a PDF: compiling every build
flavour, preparing data, pre-training, SFT, LoRA, GRPO, GPU acceleration, the
Google Colab workflow, and linking the engine into your own C program. Every
command-line parameter is documented with its default, its effect, and when you
would actually reach for it.

```sh
cd tools/guidepdf
go run .            # -> ../../build/guide.pdf
```

Go lives at `D:\env\go_lang` on this machine. The only dependency is
`github.com/jung-kurt/gofpdf`.

## How it is put together

- `doc.go` is a thin typesetting layer over gofpdf: headings with automatic
  section numbering, paragraphs, code blocks, tables, callouts, and the
  contents page.
- `main.go` is the content, one function per chapter.

**Two passes.** The first pass records which page each heading landed on; the
second renders again with a contents page built from those numbers. Both passes
reserve exactly `tocPage` pages for the contents, so the numbering agrees
between them. The console output echoes the recorded contents, which makes a
broken two-pass run (every entry on page 0, or all on one page) obvious without
opening the PDF.

## Two things to know before editing

**Stay ASCII.** gofpdf's built-in fonts are cp1252. An em dash, a curly quote or
an arrow pasted from a terminal silently becomes mojibake rather than failing
loudly. Write `->` and `-`.

**Table widths must sum to `bodyW` (174mm).** Nothing enforces it; a row that
over-runs simply collides with the next column.

## When you change the engine

This document describes real behaviour, including several unflattering
measurements. If you change a default, add a flag, or re-measure performance,
update the matching table here too - a manual that quietly drifts out of date is
worse than no manual.
