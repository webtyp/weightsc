---
PLAN: "feat!: weightsc converts Qwen3.5 — int8-block32 quantization, tensor prefix filter, added/special tokens, 3-D tensors"
TAG: v0.2.0
EXECUTOR: jules
REVIEWER: none
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.
>
> Part of
> [`AGENT_ECOSYSTEM_MASTER_PLAN.md`](https://github.com/webtyp/agent/blob/main/docs/AGENT_ECOSYSTEM_MASTER_PLAN.md).
> Depends on the published tag `webtyp.com/weights` v0.2.0 (adds `weights.Int8Block32`,
> `weights.BlockSize`, `Tensor.DequantRow`).

# Plan — `webtyp/weightsc`: convert Qwen3.5-0.8B into a browser artifact

## 0. Context

`weightsc` is a host-only command-line tool. It reads a model's original files (`config.json`,
`tokenizer.json`, `model.safetensors`) and writes a `.wtypw` artifact (the `webtyp/weights`
format the browser reads) plus a `.merges` file (the BPE merge rules). Today it only handles the
embedding model `bekko-embedding-v1-a8m`. The first language model webtyp runs,
**Qwen3.5-0.8B**, breaks it in five places, all measured on the real files:

| # | Qwen3.5 fact | What happens today |
|---|---|---|
| 1 | `vocab_size` (248 320) is inside `text_config`, not at the top of `config.json` | read as 0; the vocabulary size falls back to the BPE map (248 044) |
| 2 | 26 special tokens (`<\|endoftext\|>` = 248044, `<\|im_start\|>`, `<\|im_end\|>`, …, up to 248069) are in `tokenizer.json`'s `added_tokens`, not in `model.vocab` | lost; the chat template cannot be encoded |
| 3 | ids 248070–248319 have no token at all (padding rows of the embedding table) | with the right size, `parseVocab` fails with "vocab has no token for id 248070" |
| 4 | the file also contains a vision tower (`model.visual.*`) and a multi-token-prediction head (`mtp.*`) the text model does not use | converted anyway: ~1 GB of useless tensors |
| 5 | `linear_attn.conv1d.weight` is 3-D (`[6144][1][4]`) | `unsupported tensor shape dimension 3` |

It also needs a finer quantization. Per-row int8 loses too much precision for a model that
generates text token by token, so language models use **int8 with one scale per block of 32
values** (`weights.Int8Block32`, the GGUF `Q8_0` layout).

`bekko`'s conversion must produce the **same bytes as before** with the default options.

## Development rules (inline)

- This is **backend tooling**: it legitimately uses the standard library (`os`, `encoding/json`,
  `fmt`, `sort`, `strings`, `math`). Do **not** "fix" those imports. The browser-only rules of the
  ecosystem do not apply here.
- `cmd/weightsc/main.go` only parses flags, calls the library and prints. Every check and every
  decision is an exported library function.
- Flag names and quantization names are constants, never repeated literals.
- Tests: `testing` only, fixtures built in Go (like `createSyntheticSafetensors`), no network,
  no real model files. Do **not** run `gopush`/`codejob`.

## Design gate (api-design — five answers)

1. **Prior art.** **llama.cpp `convert_hf_to_gguf.py`** reads `text_config` for multimodal
   checkpoints, adds `added_tokens`, pads the vocabulary to the embedding size with
   `[PAD<id>]` tokens, skips vision/MTP tensors for the text model, and chooses the type with
   `--outtype q8_0|f32|…`. **Hugging Face `optimum` export** takes a quantization mode flag the
   same way. We follow llama.cpp's choices, and use our format's type names for the flag values.
2. **Novice-name test.** `-quant int8-block32` and `-prefix model.language_model.` read as what
   they do. The library gets `weightsc.Options{ID, Version, Quant, Prefix}` and
   `weightsc.Convert(inDir, opts)`. Positional `(id, version)` arguments with more options added
   would become a boolean/position soup.
3. **Complexity ledger.**
   ```
   Concepts the developer must learn   +2 (Quant, Prefix) / −0
   Lines at the call site              +1 / −0   (Options struct)
   Ways to do the same thing           +0 / −0   (one Convert, options with defaults = today's behaviour)
   ```
4. **Where it belongs.** Turning a checkpoint into the browser format is this tool's only job.
   Model-specific runtime knowledge (chat template, layer math) stays out: this tool only knows
   files, tensors, tokens and quantization.
5. **What it deletes.** The `Convert(inDir, artifactID, version)` signature (replaced by
   `Convert(inDir, Options)`), and the hard error for ≥ 3-D tensors.

## Stage 1 — options (`options.go`)

```go
// Quant is how 2-D tensors are stored in the artifact.
type Quant string

const (
	QuantInt8Row     Quant = "int8-row"     // one scale per row (default; the embedding model)
	QuantInt8Block32 Quant = "int8-block32" // one scale per 32 values (language models)
	QuantFloat32     Quant = "float32"      // no quantization (verification, small models)
)

// Options configures one conversion.
type Options struct {
	ID      string // artifact ID, required
	Version uint32 // artifact version, required (> 0)
	Quant   Quant  // "" means QuantInt8Row
	Prefix  string // when set, only tensors whose name starts with Prefix are converted (names are kept whole)
}
```

`func (o Options) Validate() error`: empty `ID` → `weightsc: Options.ID is required`;
`Version == 0` → `weightsc: Options.Version must be greater than zero`; an unknown `Quant` →
`weightsc: unknown quantization "<q>" (want int8-row, int8-block32 or float32)`.

## Stage 2 — vocabulary (`convert.go`)

- **Vocabulary size:** `vocab_size` at the top of `config.json`, or else `text_config.vocab_size`.
  If both are missing, use the largest token id + 1.
- **Added tokens:** after `model.vocab`, add every entry of `tokenizer.json`'s top-level
  `added_tokens` array (`{"id": <int>, "content": <string>}`) at its id. If an id is already
  taken by a different string, return `weightsc: token id <id> is both "<a>" and "<b>"`.
- **Padding:** every id below the vocabulary size still without a token gets `<|pad_<id>|>`
  (like llama.cpp's `[PAD<id>]`). The old "vocab has no token for id" error is removed. A
  **missing id in the middle of `model.vocab`** is still padded, and the test in Stage 5 pins that
  behaviour. Update `TestParseVocab_GapReturnsError`, which asserted the old error: rename it
  `TestParseVocab_GapIsPadded`.

## Stage 3 — tensors (`convert.go`, `quantize.go`)

- Tensors are visited in sorted name order (as today). When `Prefix != ""`, skip every name that
  does not start with it.
- **2-D tensors:** stored by `Quant`:
  - `int8-row`: today's `QuantizeRowInt8`, `weights.Int8` (bytes must not change).
  - `int8-block32`: new `QuantizeBlocksInt8(row []float32) ([]int8, []float32)` in `quantize.go`.
    For each block of `weights.BlockSize` values (the last block may be shorter),
    `scale = max|x| / 127` (a zero block gets scale 1), `q = round(x / scale)` clamped to
    [−127, 127]. Stored as `weights.Int8Block32`, with scales row by row, block by block.
  - `float32`: little-endian float32, `weights.Float32`.
- **1-D and ≥ 3-D tensors:** always float32, shape kept as in the file (for example `[6144 1 4]`).
- The source dtypes `BF16` and `F32` are already read by `ReadTensorFloat32`. Do not change it.

## Stage 4 — command line (`cmd/weightsc/main.go`)

Add `-quant` (default `int8-row`) and `-prefix` (default empty). The usage line becomes:
`weightsc -in <dir> -out <file.wtypw> -merges-out <file.merges> -id <artifact-id> -version <uint32> [-quant int8-row|int8-block32|float32] [-prefix <tensor name prefix>]`.
It builds `weightsc.Options`, calls `Validate`, then `Convert`. Errors go to stderr, and the exit
code is 1.

## Stage 5 — tests (`convert_test.go`, `quantize_test.go`)

| Test | Asserts |
|---|---|
| `TestQuantizeBlocksInt8_RoundTrip` | 40 random values → 2 scales; dequantizing each value is within `scale/2` of the original |
| `TestQuantizeBlocksInt8_ZeroBlock` | a block of zeros → scale 1, all q = 0 |
| `TestConvert_Int8Block32ThroughWeights` | a synthetic 2 × 40 tensor converted with `QuantInt8Block32` → `weights.Open` gives `Int8Block32`, 4 scales, and `DequantRow` within tolerance |
| `TestConvert_DefaultIsUnchanged` | the existing synthetic fixture converted with `Options{ID, Version}` equals, byte for byte, the output of the same conversion written as `Quant: QuantInt8Row` (and `TestConvert_RoundTripThroughWeights` keeps passing unchanged) |
| `TestConvert_PrefixFilters` | fixture with `model.language_model.a`, `model.visual.b`, `mtp.c` and `Prefix: "model.language_model."` → the artifact has only `model.language_model.a` |
| `TestConvert_ThreeDimensionalIsFloat32` | a `[3][1][4]` tensor → `weights.Float32`, shape `[3 1 4]`, values exact |
| `TestConvert_TextConfigVocabAndAddedTokens` | `config.json` = `{"text_config":{"vocab_size":8}}`, `model.vocab` with ids 0–4, `added_tokens` with id 5 `"<\|im_start\|>"` → vocab length 8, id 5 is `<\|im_start\|>`, ids 6 and 7 are `<\|pad_6\|>`, `<\|pad_7\|>` |
| `TestConvert_AddedTokenConflict` | an added token reusing an id with a different string → the conflict error |
| `TestOptions_Validate` | each of the three messages |

## Stage 6 — docs

`README.md`: document the two new flags with the exact Qwen3.5 command:

```bash
weightsc -in ~/Dev/LMmodels/Qwen/Qwen3.5-0.8B -out qwen3.5-0.8b.wtypw \
         -merges-out qwen3.5-0.8b.merges -id qwen3.5-0.8b -version 1 \
         -quant int8-block32 -prefix model.language_model.
```

State what it skips (vision tower, MTP head), and that special tokens and padding are
included. Keep the `bekko` command as the default example.

## Stages

| Stage | Files | Acceptance |
|---|---|---|
| 1 | `options.go` | `TestOptions_Validate` |
| 2 | `convert.go` | vocab tests |
| 3 | `convert.go`, `quantize.go` | tensor tests; `TestConvert_DefaultIsUnchanged` |
| 4 | `cmd/weightsc/main.go` | `go run ./cmd/weightsc` with no args prints the usage and exits 0 |
| 5 | tests | `gotest` passes |
| 6 | `README.md` | the Qwen3.5 command is present |
