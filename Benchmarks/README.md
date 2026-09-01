# Benchmarks — baseline for mapped model loading

Today `whisper_init_from_file_with_params` reads the model through
`whisper_model_loader`'s `read()` callback into freshly allocated Metal
buffers. The weights end up in anonymous, dirty memory: macOS can only reclaim
it by writing ~1.6 GB to swap, and every dictation after an idle stretch pages
it back. Mapping the file instead makes those pages clean and file-backed —
free to evict, re-read straight from the model on demand.

This directory measures the before, so the after is comparable.

## Running

```bash
./run.sh                 # probe + layout + benchmark
./run.sh --iters 5
MODEL=/path/to/model.bin ./run.sh
```

Everything links against the vendored `../lib/*.a`. No extra dependencies.

## What each tool is for

### `alignment-probe.c`

Answers two questions on the machine it runs on, by measuring rather than
trusting headers:

- does this Metal device expose `buffer_from_host_ptr` (no-copy wrapping of
  host memory), and does it really return the same pointer;
- how far can a tensor address be off before compute goes wrong.

The second one is the reason this file exists. The buffer type advertises
32-byte alignment, but that governs ggml's own allocator — the mapped loader
assigns addresses directly and bypasses it. Measured on M1 Pro:

```
offset%32    status   max |diff| vs aligned
0            ran      0.000e+00  ok
2            ran      1.439e+01  *** WRONG NUMBERS, NO ERROR ***
4            ran      0.000e+00  ok
8            ran      0.000e+00  ok
16           ran      0.000e+00  ok
30           ran      1.439e+01  *** WRONG NUMBERS, NO ERROR ***
```

**Four bytes is the floor, and violating it is silent.** No error, no assert,
no crash — the kernel runs and returns wrong values. A naive mapping would
ship an app that works and quietly transcribes worse. The loader has to refuse
a misaligned tensor and fall back, and the probe should be re-run on any new
GPU family before trusting the mapped path there.

### `model-layout.py`

Reports how a ggml model is laid out and whether it can be mapped as-is. The
stock turbo model cannot:

```
tensors : 587
   4 bytes :  113/587  ( 19.3%)  <- BLOCKS MAPPING
offsets mod 4 seen: [0, 1, 2, 3]
repack to  4-byte alignment: +889 bytes (0.00005% of file)
repack to 32-byte alignment: +8,821 bytes (0.00054% of file)
```

474 of 587 tensors are misaligned. The format writes each tensor's header
immediately before its data, so offsets land wherever the previous tensor
happened to end. Aligning them costs 8.8 KB on a 1.6 GB file — the repack is
essentially free; it is the format, not the size, that blocks mapping.

Run this again after a repack to confirm 587/587.

### `bench.c`

Load time, memory, and a correctness fingerprint, per iteration.

The `internal` / `external` split is the number to watch. `internal` is
anonymous memory — reclaimable only by swapping it out. `external` is
file-backed — droppable for free. The patch should move the weights from the
first column to the second.

The fingerprint runs the encoder and decoder on Metal over a fixed synthetic
take and hashes the logits. Since misalignment corrupts silently, a transcript
diff is too weak a check; the hash catches it directly.

## Baseline — M1 Pro, macOS 26.5.1, ggml-large-v3-turbo.bin

```
it       load_s   encode_s    footprint     internal     external        logits_hash
---    --------   --------   ----------   --dirty---   --clean---   ----------------
0         0.842      1.054      1768.5M      1748.7M        14.9M 0x779680b647c15dfc
1         0.815      1.013      1602.0M      1594.8M         0.0M 0x779680b647c15dfc
2         0.723      1.055      1603.5M      1603.2M       -11.6M 0x779680b647c15dfc

load   : mean 0.793 s   best 0.723 s
encode : mean 1.041 s
weights: mean 1658.0 MB phys_footprint
peak   : 1800.8 MB lifetime max footprint
logits : 0x779680b647c15dfc   stable across iterations
```

Loads are warm-cache — the file is in the page cache from the previous
iteration. A genuinely cold first load is slower; measuring it needs `sudo
purge`, so it is deliberately not part of the automated run.

## Targets after the patch

| metric | baseline | expected | why |
|---|---|---|---|
| `load_s` | 0.79 s | well under | no 1.6 GB read, no copy — map and assign |
| `internal` (dirty) | ~1600 MB | near zero | weights become file-backed |
| `external` (clean) | ~0 MB | ~1600 MB | the whole point |
| peak footprint | 1800 MB | lower | no staging buffer during load |
| `encode_s` | 1.04 s | unchanged | same weights, same kernels |
| logits hash | `0x779680b647c15dfc` | **identical** | anything else means corruption |

The hash is the gate. Faster loading with a changed hash is a regression, not
a win.
