#!/usr/bin/env python3
"""Rewrite a ggml whisper model so its tensor data can be mapped instead of read.

The stock format writes each tensor's header immediately before its data, so
data offsets land wherever the previous tensor happened to end — 474 of 587 in
large-v3-turbo are not even 4-byte aligned. Pointing Metal at an offset like
that does not fail; it silently computes wrong numbers.

This pads each tensor's data up to a 32-byte boundary and stamps a different
magic so the loader knows the padding is there. Nothing else changes: same
hparams, same vocab, same weights, same order. The file grows by ~9 KB.

    ./repack-model.py in.bin out.bin
"""

import os
import shutil
import struct
import sys

GGML_MAGIC = 0x67676D6C  # "ggml" — stock
MAP_MAGIC = 0x67676D6D   # "ggmm" — padded, mappable
ALIGN = 32


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]

    fi = open(src, "rb")
    magic, = struct.unpack("<I", fi.read(4))
    if magic == MAP_MAGIC:
        sys.exit(f"{src} is already repacked")
    if magic != GGML_MAGIC:
        sys.exit(f"{src} is not a ggml model (magic {magic:#x})")

    tmp = dst + ".tmp"
    fo = open(tmp, "wb")
    fo.write(struct.pack("<I", MAP_MAGIC))

    # hparams, mel filters, vocab: copied through untouched
    hp = fi.read(44)
    fo.write(hp)
    n_mel, n_fft = struct.unpack("<2i", fi.read(8))
    fo.write(struct.pack("<2i", n_mel, n_fft))
    fo.write(fi.read(n_mel * n_fft * 4))

    n_vocab, = struct.unpack("<i", fi.read(4))
    fo.write(struct.pack("<i", n_vocab))
    for _ in range(n_vocab):
        ln, = struct.unpack("<i", fi.read(4))
        fo.write(struct.pack("<i", ln))
        fo.write(fi.read(ln))

    n, padding = 0, 0
    while True:
        head = fi.read(12)
        if len(head) < 12:
            break
        n_dims, name_len, ftype = struct.unpack("<3i", head)
        if not 1 <= n_dims <= 4:
            sys.exit(f"tensor {n}: implausible n_dims {n_dims}")
        dims = fi.read(4 * n_dims)
        name = fi.read(name_len)

        fo.write(head)
        fo.write(dims)
        fo.write(name)

        pad = (-fo.tell()) % ALIGN
        fo.write(b"\0" * pad)
        padding += pad
        assert fo.tell() % ALIGN == 0

        ne = struct.unpack(f"<{n_dims}i", dims)
        count = 1
        for d in ne:
            count *= d
        nbytes = count * (2 if ftype == 1 else 4)

        left = nbytes
        while left:
            chunk = fi.read(min(left, 8 << 20))
            if not chunk:
                sys.exit(f"tensor {n} ({name.decode()}): file ended early")
            fo.write(chunk)
            left -= len(chunk)
        n += 1

    fo.close()
    fi.close()
    shutil.move(tmp, dst)

    before, after = os.path.getsize(src), os.path.getsize(dst)
    print(f"tensors  : {n}")
    print(f"padding  : {padding:,} bytes")
    print(f"size     : {before:,} -> {after:,}  (+{(after - before) / before * 100:.5f}%)")
    print(f"written  : {dst}")


if __name__ == "__main__":
    main()
