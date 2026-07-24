"""Tile P6 PPM files into one nearest-neighbour-scaled PNG (stdlib only)."""
import sys, zlib, struct

def load_ppm(path):
  with open(path, 'rb') as f:
    data = f.read()
  # header: P6 <w> <h> <maxval>, whitespace/comment separated
  fields, i = [], 2
  while len(fields) < 3:
    while i < len(data) and data[i:i+1].isspace(): i += 1
    if data[i:i+1] == b'#':
      while data[i:i+1] not in (b'\n', b''): i += 1
      continue
    j = i
    while j < len(data) and not data[j:j+1].isspace(): j += 1
    fields.append(int(data[i:j])); i = j
  i += 1  # single whitespace after maxval
  w, h = fields[0], fields[1]
  return w, h, data[i:i + w * h * 3]

def png(path, w, h, rgb):
  raw = b''.join(b'\x00' + rgb[y*w*3:(y+1)*w*3] for y in range(h))
  def chunk(tag, body):
    c = tag + body
    return struct.pack('>I', len(body)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
  with open(path, 'wb') as f:
    f.write(b'\x89PNG\r\n\x1a\n')
    f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)))
    f.write(chunk(b'IDAT', zlib.compress(raw, 9)))
    f.write(chunk(b'IEND', b''))

def main():
  out = sys.argv[1]
  cols = int(sys.argv[2])
  scale = int(sys.argv[3])
  srcs = sys.argv[4:]
  tiles = [load_ppm(p) for p in srcs]
  tw, th = tiles[0][0], tiles[0][1]
  pad = 2
  rows = (len(tiles) + cols - 1) // cols
  cw, ch = tw * scale + pad, th * scale + pad
  W, H = cols * cw + pad, rows * ch + pad
  canvas = bytearray(b'\x20' * (W * H * 3))
  for n, (w, h, px) in enumerate(tiles):
    ox = pad + (n % cols) * cw
    oy = pad + (n // cols) * ch
    for y in range(h * scale):
      sy = y // scale
      row = oy + y
      for x in range(w * scale):
        sx = x // scale
        si = (sy * w + sx) * 3
        di = (row * W + ox + x) * 3
        canvas[di:di+3] = px[si:si+3]
  png(out, W, H, bytes(canvas))
  print('wrote %s (%dx%d, %d tiles)' % (out, W, H, len(tiles)))

main()
