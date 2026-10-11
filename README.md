# Qwen3.8-Flash-Next abliterated — EXL3 3.05bpw integer recipe (DGX Spark / GB10)

Requant of [huihui-ai's Huihui-Qwen3.8-Flash-Next-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated)
to the stock 3.05bpw EXL3 allocation, built and validated on NVIDIA DGX Spark (GB10, 128GB
unified). Serving on two single-node Sparks pooled behind an OpenAI-compatible router.
This repo is the recipe, the traps, and the validation battery — the weights live on
Hugging Face (link at bottom once published).

## What this is

| | |
|---|---|
| Source | huihui-ai abliterated bf16 (LM layers abliterated; n-gram/MTP/vision byte-identical to base) |
| Target | EXL3 3.05bpw — **integer-pinned per-tensor recipe** |
| Allocation | 75,277 tensors @ 3bpw, 472 @ 5bpw, 2 @ 4bpw + head 5 / vision 5 / MTP 3 / n-gram 5 |
| Hardware | 1× DGX Spark per instance (no TP), 262,144-token context, max 5 concurrent |
| Engine | EXL3 with MTP2 + n-gram speculative decoding, k=7 |

## Why a custom recipe (the two hard traps)

**Trap 1 — the GB10 EXL3 loader rejects fractional trellis widths.** exllamav3 v1.6.x's
LP allocator emits half-integer expert rates (3.5bpw) under the mul1 codebook. A plain
`-b 3.05` conversion produces a pack the loader refuses at startup:

```
thread 'main' panicked: trellis dims 56 (bits 3.5)
```

The fix: extract the stock pack's per-tensor allocation and pin every tensor to it. The
extraction reads the safetensors headers of a stock 3.05bpw pack — for trellis tensors,
bits = last_dim / 16:

```python
# stock pack allocation -> converter recipe YAML
recipe = {}
for f in glob("stock-pack/model-*.safetensors"):
    with open(f, "rb") as fh:
        n, = struct.unpack("<Q", fh.read(8))          # header length
        hdr = json.loads(fh.read(n))
    for name, info in hdr.items():
        if name == "__metadata__": continue
        if info["dtype"].startswith("u"):              # trellis-packed
            bits = info["shape"][-1] / 16
            recipe[name] = int(bits)
yaml.safe_dump({"tensors": recipe, "achieved_bpw": 3.05, "head_bits": 5}, ...)
```

Feed it with `-rcp recipe.yaml` (v1.6.x) — skips the LP search entirely and guarantees the
pack matches the stock memory profile and speed.

**Trap 2 — `--ngram_file` doesn't splice the index.** Reusing a pre-quantized n-gram table
(32GB `ngram_embedding.safetensors`) copies the file into the output but does **not** add its
132 `ngram_embedding.*` entries to `model.safetensors.index.json`. The loader panics
(`ple shard 0 not in index`). Post-process: merge the entries from a complete pack's index
and set `metadata.total_size` to include the n-gram file (84,946,203,828 for this recipe).

**Trap 3 — PTX kernels resolve relative to CWD.** The serve binary looks for
`src/ptx/*.ptx` in its working directory. Launch from the engine tree, not `$HOME`:
`[exl3-serve] exl3_bench.ptx missing — run cargo build --release` is the symptom.

## Conversion environment (no root on the box)

Converter: exllamav3 v1.6.0 (build from source). On a sudoless GB10 host:

```bash
# Python headers without apt install
apt-get download python3.12-dev libpython3.12-dev && dpkg-deb -x *.deb ~/pyhdrs/
export CPATH=~/pyhdrs/usr/include/python3.12:$CPATH   # + multiarch pyconfig.h copy
# CUDA build needs proxy vars stripped or pip stalls silently
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY pip install torch --index-url \
  https://download.pytorch.org/whl/cu130
```

## Full convert command

```bash
python3 convert.py -i <huihui-bf16-src> -w <work> -o <out> \
  -rcp integer-recipe.yaml \
  -hb 5 -mb 3 -vb 5 -ngb 5 \
  --ngram_file <stock>/ngram_embedding.safetensors \
  --out_scales always --codebook mul1 \
  -cr 250 -cc 2048 -cpi 120
```

Runtime on one Spark, ~5–9h for the 75.8k-tensor model (checkpointed every 120s;
`--resume` recovers). ~26GB RSS — runs alongside a live serve if memory allows.

## Validation battery (all green)

| Gate | Method | Result |
|---|---|---|
| Refusal triage | 20-prompt rapid protocol, temp 0.3 | **20/20 answered, 0 refusals** |
| Speed | usage-token math, thinking-off, median of 6 | **65 tok/s** (stock ~54 idle) |
| MTP2 | serve-log acceptance stats | 52–63%, 2.26–3.87 tok/round, k=7 |
| Long context | 226,497-token needle, mid-haystack | **recovered** (~1,560 tok/s prefill) |
| Vision | inline red-square PNG probe | correct |
| Determinism | temp-0 double-run | identical |
| Tool calls | get_weather round-trip | well-formed JSON args |
| Streaming / concurrency | SSE chunks / 4-way parallel | 72 chunks / 1.4s wall |

## Deployment shape

One Spark per instance: the full 84GB pack + 262k context fit a single GB10 with no tensor
parallelism. Two independent serves pooled behind an OpenAI-compatible router = 10 full-context
concurrent slots. Batching is load-adaptive (speculation collapses toward plain steps as slots
fill — throughput trades per-slot speed for full-context concurrency; hybrid GDN arch keeps
per-token KV to 12 of 48 layers).

## Provenance

- Base: Qwen/Qwen3.8-Flash-Next (Qwen Community License 1.0).
- Abliteration: huihui-ai (their card carries usage warnings + donation links).
- Requant: this recipe — per-tensor integer allocation pinned to the stock pack, n-gram/MTP/
  vision sidecars carried through unmodified.

Thanks to [tonyd2wild](https://github.com/tonyd2wild) — the Flash-Next fleet-deploy
foundation this lane descends from.
