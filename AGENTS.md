# Agent Guide — `webtyp/weightsc`

Constraints for agents working on this repo. **Read this before any change.**
The current work order is [docs/PLAN.md](docs/PLAN.md); the master index is
[`agent/docs/MASTER_PLAN.md`](https://github.com/webtyp/agent/blob/main/docs/MASTER_PLAN.md).

---

## What this repo is

A host-only CLI: reads a HuggingFace `safetensors` model plus its `config.json` and
`tokenizer.json`, and writes the `WTYPW1` artifact that [`webtyp/weights`](https://github.com/webtyp/weights)
reads, plus a companion `.merges` file for [`webtyp/tokenizer`](https://github.com/webtyp/tokenizer).
It is a **build-time tool**, never shipped to a browser.

---

## This repo does NOT compile to WASM. Stdlib is legitimate here.

Most of this ecosystem bans `os`, `flag`, `net/http`, `encoding/json`, `fmt`, `errors`,
`map[K]V` in code that ships to a browser tab under TinyGo. **None of that applies here.**
This repo never runs under `tinygo build -target wasm` and never will — it is the same
category as `webtyp/ormc`, `webtyp/ddlc`, `webtyp/sitec`. If you find yourself "fixing" a
stdlib import in this repo because you saw the rule elsewhere in the ecosystem, stop: that
rule protects a WASM binary size/compatibility budget this repo doesn't have.

The only import boundary that matters here: [`webtyp.com/weights`](https://github.com/webtyp/weights)
for `WriteArtifact` and its types. Don't reimplement the binary format — call it.

---

## The build that defines "done"

```bash
go vet ./...
gotest
```

No `GOOS=js GOARCH=wasm`, no `tinygo test`, no `tinygo build` — none apply to a host-only CLI.

---

## Layout & tests

- `convert.go` (or split by domain if it grows past 500 lines) holds all logic: safetensors
  parsing, BF16→float32, quantization, artifact + merges-file writing. Testable without a CLI.
- `cmd/weightsc/main.go` is thin: flag parsing, call into the library, `os.Exit`. No logic.
- Publish with `gopush 'message'` — never `git commit`/`git push` directly.

## Common mistakes to avoid

- Treating this as a general-purpose model converter. It converts exactly the ModernBERT
  tensor layout `docs/PLAN.md` documents, for the one model this project needs. Generalizing
  ahead of a second model is speculative work nobody asked for.
- Quantizing the 1D `*_norm.weight` tensors. Only 2D weight matrices get int8 + per-row scale.
- Splitting the fused `Wqkv` / `mlp.Wi` tensors. That's `webtyp/transformer`'s decision at load
  time, not this repo's.
- Adding a field to `weights.TokenizerConfig` to carry BPE merges. That struct is published
  API in a different repo — `docs/PLAN.md` already resolved this by writing a separate
  `.merges` file instead. Don't reopen it here.
