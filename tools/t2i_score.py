"""Objectively score t2i two-object scenes on two SEPARATE axes.

Eyeballing a montage conflates two different failures, so they are measured apart:

  color fidelity  -- are both requested colors actually present as blobs?
  relation acc    -- GIVEN both present, is object1 on the side the caption states?

Pixels are assigned to the nearest palette color (palette must match T2I_COLORS in
t2i.c); a color counts as present only if it owns >= MIN_PX pixels, so anti-aliased
fringe pixels are not read as a blob.

  usage: t2i_score.py <manifest.tsv> [--dump]
  manifest lines (tab separated): <ppm-path>  <color1>  <color2>  <relation>

CALIBRATION: always score the ground-truth `data/<set>/img_*.ppm` too. Those set the
measurement ceiling -- they read 100% color fidelity but only ~88% relation accuracy
(jittered shapes can straddle), so ~88% is a PASS for relations, not a shortfall.
"""
import sys, collections

PAL = {
  'red': (217, 31, 31), 'green': (33, 178, 51), 'blue': (38, 76, 229),
  'yellow': (242, 217, 38), 'cyan': (38, 204, 217), 'magenta': (217, 51, 204),
  'orange': (242, 140, 26), 'purple': (140, 51, 191), 'bg': (20, 20, 20),
}
MIN_PX = 8

def load_ppm(path):
  with open(path, 'rb') as f:
    data = f.read()
  fields, i = [], 2
  while len(fields) < 3:
    while data[i:i+1].isspace(): i += 1
    j = i
    while not data[j:j+1].isspace(): j += 1
    fields.append(int(data[i:j])); i = j
  i += 1
  w, h = fields[0], fields[1]
  return w, h, data[i:i + w*h*3]

def centroids(w, h, px):
  acc = collections.defaultdict(lambda: [0.0, 0.0, 0])
  for y in range(h):
    for x in range(w):
      p = (y*w + x)*3
      r, g, b = px[p], px[p+1], px[p+2]
      best, bd = None, 1e18
      for name, (pr, pg, pb) in PAL.items():
        d = (r-pr)**2 + (g-pg)**2 + (b-pb)**2
        if d < bd: bd, best = d, name
      if best == 'bg': continue
      a = acc[best]; a[0] += x; a[1] += y; a[2] += 1
  return {k: (v[0]/v[2], v[1]/v[2], v[2]) for k, v in acc.items() if v[2] >= MIN_PX}

def main():
  dump = '--dump' in sys.argv
  rows = [l.lstrip('﻿').rstrip('\n').split('\t')
          for l in open(sys.argv[1], encoding='utf-8-sig') if l.strip()]
  rel_per = collections.defaultdict(lambda: [0, 0])
  col_ok = col_n = 0
  for path, c1, c2, rel in rows:
    w, h, px = load_ppm(path)
    cen = centroids(w, h, px)
    col_n += 1
    both = c1 in cen and c2 in cen
    col_ok += both
    if dump:
      blobs = ' '.join('%s:%d' % (k, v[2]) for k, v in
                       sorted(cen.items(), key=lambda kv: -kv[1][2]))
      print('%-14s want %s+%s %-9s got [%s]%s' %
            (path.split('\\')[-1], c1, c2, rel, blobs, '' if both else '  <-- MISSING'))
    if not both: continue
    x1, y1, _ = cen[c1]; x2, y2, _ = cen[c2]
    ok = {'left of':  x1 < x2, 'right of': x1 > x2,
          'above':    y1 < y2, 'below':    y1 > y2}[rel]
    rel_per[rel][0] += ok
    rel_per[rel][1] += 1
  print('\ncolor fidelity (both requested colors present): %.0f%%  %d/%d' %
        (100.0*col_ok/col_n, col_ok, col_n))
  print('\nrelation accuracy, among scenes where both colors rendered:')
  tot = [0, 0]
  for rel in ('left of', 'right of', 'above', 'below'):
    ok, n = rel_per[rel]
    tot[0] += ok; tot[1] += n
    print('  %-9s %s' % (rel, ('%3.0f%%  %d/%d' % (100.0*ok/n, ok, n)) if n else '   -   0/0'))
  if tot[1]:
    print('  %-9s %3.0f%%  %d/%d   (chance = 50%%)' %
          ('OVERALL', 100.0*tot[0]/tot[1], tot[0], tot[1]))

main()
