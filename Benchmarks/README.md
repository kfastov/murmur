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

## Result after the patch

Same machine, same model weights, `--iters 3`. "footprint" is what the process
is charged for — the number Activity Monitor shows and the one memory pressure
acts on; "resident" is what is actually in RAM.

```
stock libs + stock model
it       load_s   encode_s    footprint     resident        logits_hash
0         0.984      1.120      1782.5M      1763.2M 0x779680b647c15dfc
1         0.809      1.028      1608.8M      1603.7M 0x779680b647c15dfc
peak   : 1794.6 MB

patched libs + repacked model
it       load_s   encode_s    footprint     resident        logits_hash
0         0.960      1.047       100.5M        86.8M 0x779680b647c15dfc
1         0.356      1.036        63.7M        68.7M 0x779680b647c15dfc
peak   :  110.4 MB
```

| metric | stock | mapped | |
|---|---|---|---|
| footprint after use | 1695 MB | **82 MB** | 20x |
| resident | 1604 MB | **69 MB** | 23x |
| peak during load | 1795 MB | **110 MB** | 16x |
| load, warm | 0.81 s | **0.36 s** | 2.2x |
| encode | 1.03 s | 1.04 s | unchanged |
| logits hash | `0x779680b647c15dfc` | `0x779680b647c15dfc` | **identical** |

The gate passes: the fingerprint is bit-identical, so the mapped weights are
being read correctly.

The weights turn out not to be resident at all. Metal reads them through the
mapping as the graph runs, and because the pages are clean the kernel drops
them again instead of paying to keep them — so an idle model costs nothing and
never reaches swap. Encode time is unchanged, so on this machine there is no
measurable price for re-reading.

Iteration 0 is consistently slower in both columns: that is the page cache
filling, not the loader.

## Applying it

The patch is against whisper.cpp **v1.8.4**, which is the version the vendored
`lib/*.a` were built from (all three public headers match that tag exactly).

```bash
git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp
git checkout v1.8.4
git apply ../Patches/0001-whisper-map-model-weights.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
      -DGGML_METAL_EMBED_LIBRARY=ON -DBUILD_SHARED_LIBS=OFF \
      -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_TESTS=OFF
cmake --build build -j
# copy build/src/libwhisper.a and build/ggml/src/**/libggml*.a into ../lib/
```

Then repack the model once:

```bash
./repack-model.py model.bin model.mapped.bin
```

155 lines in one file, 116 of them code. Nothing is added to the read path:
a stock model still loads exactly as before, and a repacked one falls back to
reading if the mapping or the backend capability is unavailable.
