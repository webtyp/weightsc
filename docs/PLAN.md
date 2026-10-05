---
PLAN: "feat: -quant int4-block32 — convert checkpoints to 4-bit blocks (weights.Int4Block32)"
TAG: v0.3.0
EXECUTOR: jules
REVIEWER: none
STATUS: running
SESSION: 12465041048023580988
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.


# Plan — `weightsc` v0.3.0: 4-bit conversion

**Read [AGENTS.md](../AGENTS.md) first: this repo is a host-only CLI; the standard library is
correct here.** Master plan:
[AGENT_ECOSYSTEM_MASTER_PLAN.md](https://github.com/webtyp/agent/blob/main/docs/AGENT_ECOSYSTEM_MASTER_PLAN.md)
(D13: 4-bit blocks).

## Why

`webtyp.com/weights` v0.4.0 adds `Int4Block32` (GGUF Q4_0 layout with float32 scales) and its
quantizer `weights.QuantizeInt4Block32(row []float32) (q []byte, scales []float32, err error)`
(returns `weights.ErrInt4Cols` when `len(row)` is not a multiple of 32). This converter must write
it, so decider-0.8b (851 MB in int8) becomes ≈ 530 MB and fits the clinic's 4 GB machines.

## Design gate

1. **Prior art.** llama.cpp `llama-quantize … Q4_0`: every 2-D weight to Q4_0; a matrix whose row
   length is not a multiple of the block size falls back to a wider type.
2. **Novice-name test.** `-quant int4-block32`, `QuantInt4Block32`, next to `int8-block32`.
3. **Complexity ledger.** +1 value of an existing flag and constant.
4. **Where it belongs.** Here; the quantizer itself is `weights`' (the format owner), not copied.
5. **What it deletes.** Nothing.

## Stage 1 — `go.mod`

`go get webtyp.com/weights@v0.4.0`, `go mod tidy`.

## Stage 2 — `options.go`

```go
	QuantInt4Block32 Quant = "int4-block32" // 4 bits per value, one scale per 32 (GGUF Q4_0); rows not a multiple of 32 stay int8-block32
```

`Validate` accepts it; the error message becomes
`weightsc: unknown quantization %q (want int8-row, int8-block32, int4-block32 or float32)`.

## Stage 3 — `convert.go`

In the `switch quant` for 2-D tensors, a new case:

```go
case QuantInt4Block32:
	if cols%weights.BlockSize != 0 {
		// falls back to the QuantInt8Block32 branch for this tensor (same code, factored into a helper)
	}
	data := make([]byte, 0, rows*cols/2)
	scales := make([]float32, 0, rows*cols/weights.BlockSize)
	for r := 0; r < rows; r++ {
		q, s, err := weights.QuantizeInt4Block32(f32s[r*cols : (r+1)*cols])
		if err != nil { return nil, nil, fmt.Errorf("tensor %s: %w", name, err) }
		data = append(data, q...)
		scales = append(scales, s...)
	}
	inputs = append(inputs, weights.TensorInput{Name: name, DType: weights.Int4Block32, Shape: shape, Data: data, Scales: scales})
```

Factor the existing `QuantInt8Block32` branch into `func int8Block32Input(name string, shape []int, f32s []float32) weights.TensorInput` so both cases call it (the fallback must not duplicate it).

## Stage 4 — `cmd/weightsc/main.go`

Usage line and `-quant` help list `int4-block32`.

## Stage 5 — tests (`convert_test.go`, like `TestConvert_Int8Block32ThroughWeights`)

| Test | Proves |
|---|---|
| `TestConvert_Int4Block32ThroughWeights` | the existing fixture converted with `Quant: QuantInt4Block32` → `weights.Open` → every 2-D tensor whose cols are a multiple of 32 is `Int4Block32`, its `DequantRow` within half a step of the original float values (`scale/2 + 1e-6` per block); 1-D tensors stay float32 |
| `TestConvert_Int4FallsBackToInt8` | a 2-D tensor with 40 columns (build a tiny safetensors in the test, as other tests do) → `Int8Block32` |
| `TestOptions_Validate` (existing, extended) | `int4-block32` accepted; the unknown-value message lists it |

## Stage 6 — docs

`README.md`: the quantization table gets `int4-block32` (size ≈ 0.625 bytes per value; when to use
it: language models for the browser). Example:
`weightsc -in <dir> -out decider-0.8b.q4.wtypw -merges-out decider-0.8b.merges -id decider-0.8b -version 1 -quant int4-block32 -prefix model.language_model.`

## Acceptance

- `gotest` green. Never run `gopush` or `codejob`.
- `grep -n "QuantizeInt4Block32" *.go` shows the call in `convert.go` and no local copy of the algorithm.

| Stage | Files | Done when |
|---|---|---|
| 1 | `go.mod` | weights v0.4.0 |
| 2 | `options.go` | constant + validation |
| 3 | `convert.go` | int4 case + fallback helper |
| 4 | `cmd/weightsc/main.go` | usage |
| 5 | `convert_test.go` | table green |
| 6 | `README.md` | documented |
