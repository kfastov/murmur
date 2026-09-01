#!/usr/bin/env python3
"""Report how a ggml whisper model is laid out on disk.

The mapped loader points tensors straight at file offsets, so those offsets
have to be aligned. Stock models from HuggingFace are not: the format writes
each tensor's header immediately before its data, so offsets land wherever the
previous tensor happened to end.

Run this on a model before mapping it, and again after a repack to confirm the
repack did what it claims.

    ./model-layout.py ~/Library/Application\\ Support/Murmur/models/ggml-large-v3-turbo.bin
"""

import os
import struct
import sys

HPARAMS = ("n_vocab n_audio_ctx n_audio_state n_audio_head n_audio_layer "
           "n_text_ctx n_text_state n_text_head n_text_layer n_mels ftype").split()


def read_tensors(path):
    f = open(path, "rb")
    magic = f.read(4)
    if struct.unpack("<I", magic)[0] != 0x67676D6C:
        sys.exit(f"not a ggml model: magic {magic!r}")

    hp = struct.unpack("<11i", f.read(44))
    n_mel, n_fft = struct.unpack("<2i", f.read(8))
    f.seek(n_mel * n_fft * 4, 1)                      # mel filters
    n_vocab, = struct.unpack("<i", f.read(4))
    for _ in range(n_vocab):                          # vocab
        ln, = struct.unpack("<i", f.read(4))
        f.seek(ln, 1)

    first_header = f.tell()
    tensors = []
    while True:
        head = f.read(12)
        if len(head) < 12:
            break
        n_dims, name_len, ftype = struct.unpack("<3i", head)
        if not 1 <= n_dims <= 4:
            break
        ne = struct.unpack(f"<{n_dims}i", f.read(4 * n_dims))
        name = f.read(name_len).decode("utf-8", "replace")
        offset = f.tell()
        count = 1
        for d in ne:
            count *= d
        nbytes = count * (2 if ftype == 1 else 4)
        tensors.append((name, offset, nbytes))
        f.seek(nbytes, 1)
    return dict(zip(HPARAMS, hp)), first_header, tensors, f.tell()


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    hp, first_header, tensors, end = read_tensors(path)
    size = os.path.getsize(path)

    print(f"file    : {path}")
    print(f"size    : {size:,} bytes")
    print(f"decoder : {hp['n_text_layer']} layers   encoder: {hp['n_audio_layer']} layers"
          f"   ({'turbo/distilled' if hp['n_text_layer'] <= 8 else 'full'})")
    print(f"tensors : {len(tensors)}   data ends at {end:,}"
          f"{'  (trailing bytes!)' if end != size else ''}\n")

    print("alignment of tensor data offsets:")
    for a in (2, 4, 8, 16, 32):
        ok = sum(1 for _, o, _ in tensors if o % a == 0)
        flag = ""
        if a == 4:
            flag = "  <- mappable" if ok == len(tensors) else "  <- BLOCKS MAPPING"
        print(f"  {a:2d} bytes : {ok:4d}/{len(tensors)}  ({ok / len(tensors) * 100:5.1f}%){flag}")

    residues = sorted({o % 4 for _, o, _ in tensors})
    print(f"\noffsets mod 4 seen: {residues}")

    # what a repack would cost: header, then pad, then data
    f = open(path, "rb")
    for align in (4, 32):
        cur, pad = first_header, 0
        for name, off, nbytes in tensors:
            f.seek(off)
            # header size is recoverable from the gap to the previous data end
            cur += 12 + 4 * 1 + len(name)   # lower bound; exact value re-derived below
        # exact: walk gaps
        cur, pad = first_header, 0
        prev_end = None
        for i, (name, off, nbytes) in enumerate(tensors):
            header = off - (prev_end if prev_end is not None else first_header)
            cur += header
            if cur % align:
                pad += align - (cur % align)
                cur += align - (cur % align)
            cur += nbytes
            prev_end = off + nbytes
        print(f"repack to {align:2d}-byte alignment: +{pad:,} bytes "
              f"({pad / size * 100:.5f}% of file)")

    unaligned = [(n, o) for n, o, _ in tensors if o % 4]
    if unaligned:
        print(f"\n{len(unaligned)} tensors are not 4-byte aligned, e.g.:")
        for n, o in unaligned[:4]:
            print(f"  {n:36s} off={o:>13,}  mod4={o % 4}")


if __name__ == "__main__":
    main()
