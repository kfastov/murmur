#!/usr/bin/env bash
# Build and run the mmap baseline harness against the libs in ../lib.
#
#   ./run.sh                 # probe + benchmark with the default model
#   ./run.sh --iters 5       # extra passes
#   MODEL=/path/to.bin ./run.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
OUT="$HERE/.build"
MODEL="${MODEL:-$HOME/Library/Application Support/Murmur/models/ggml-large-v3-turbo.bin}"

LIBS=(
  "$ROOT/lib/libggml.a"
  "$ROOT/lib/libggml-base.a"
  "$ROOT/lib/libggml-metal.a"
  "$ROOT/lib/libggml-cpu.a"
  "$ROOT/lib/libggml-blas.a"
)
FRAMEWORKS=(-framework Foundation -framework Metal -framework MetalKit -framework Accelerate)

# whisper.cpp itself is not in lib/ — only ggml is. The benchmark needs it, so
# it is optional: the probe runs regardless.
WHISPER_LIB=""
for cand in "$ROOT/lib/libwhisper.a" "$ROOT/.build/libwhisper.a"; do
  [[ -f "$cand" ]] && WHISPER_LIB="$cand" && break
done

mkdir -p "$OUT"

echo "==> building alignment-probe"
clang -O2 -I"$ROOT/include" "$HERE/alignment-probe.c" "${LIBS[@]}" -lc++ "${FRAMEWORKS[@]}" -o "$OUT/alignment-probe"

echo "==> Metal capability + alignment floor"
echo
"$OUT/alignment-probe" 2>/dev/null || echo "(probe reported a problem — see above)"
echo

if [[ -f "$MODEL" ]]; then
  echo "==> model layout on disk"
  echo
  python3 "$HERE/model-layout.py" "$MODEL"
  echo
else
  echo "!! model not found at: $MODEL"
  echo "   set MODEL=... to point at one"
  exit 0
fi

if [[ -z "$WHISPER_LIB" ]]; then
  cat <<'EOF'
==> load benchmark skipped

bench.c needs libwhisper.a, which this repo does not vendor (lib/ holds ggml
only; whisper itself is linked into the app from elsewhere). Build whisper.cpp
once and drop libwhisper.a into lib/, then re-run — the same build is needed
for the mmap patch anyway.
EOF
  exit 0
fi

echo "==> building bench"
clang -O2 -I"$ROOT/include" "$HERE/bench.c" "$WHISPER_LIB" "${LIBS[@]}" -lc++ "${FRAMEWORKS[@]}" -o "$OUT/bench"

echo "==> load time / memory / logits fingerprint"
echo
"$OUT/bench" --model "$MODEL" "$@"
