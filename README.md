# weightsc
<img src="docs/img/badges.svg">

offline safetensors to WTYPW1 artifact converter

Documentación:
- [`docs/PLAN.md`](docs/PLAN.md) — el plan de trabajo, formato de origen y de destino
- [`docs/TIPOS_NUMERICOS_WASM.md`](docs/TIPOS_NUMERICOS_WASM.md) — qué tipos numéricos existen
  realmente en WASM, y por qué este repo convierte BF16 e int8 (para devs junior)

## Usage

### Default (Embedding Models e.g. bekko)

```bash
weightsc -in path/to/bekko -out bekko.wtypw -merges-out bekko.merges -id bekko-embedding-v1-a8m -version 1
```

### Language Models (e.g. Qwen3.5-0.8B)

Language models require finer quantization (`int8-block32`) and tensor prefix filtering to skip vision/MTP weights:

```bash
weightsc -in ~/Dev/LMmodels/Qwen/Qwen3.5-0.8B -out qwen3.5-0.8b.wtypw \
         -merges-out qwen3.5-0.8b.merges -id qwen3.5-0.8b -version 1 \
         -quant int8-block32 -prefix model.language_model.
```

The input directory may hold one `model.safetensors` or, as Hugging Face publishes larger
models, shards listed in `model.safetensors.index.json` (Qwen3.5-0.8B ships as
`model.safetensors-00001-of-00001.safetensors`). The result for Qwen3.5-0.8B: 851 MB, 320 tensors
(187 `int8b32`, 133 `float32`), vocabulary 248 320 with the special tokens at their ids.

This skips non-text model weights (such as vision tower `model.visual.*` or MTP head `mtp.*`) and ensures special tokens and vocabulary padding are included in the output artifact.
