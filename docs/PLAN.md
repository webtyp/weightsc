---
PLAN: "fix: weightsc — version bounds, sparse-vocab guard, go.mod tidy, flag help order, bit-exact bf16 test"
TAG: v0.1.1
EXECUTOR: jules
REVIEWER: none
STATUS: review
SESSION: 6232063564342468673
PR: https://github.com/webtyp/weightsc/pull/2
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.
> Índice maestro: https://github.com/webtyp/agent/blob/main/docs/MASTER_PLAN.md.
>
> **Este repositorio es host-only, no compila a WASM.** `os`, `flag`, `map[K]V` y el resto de
> la stdlib de Go son legítimos acá — ver `AGENTS.md` de este repo, que lo fija explícitamente.
> Si alguna regla de otro repo del ecosistema pareciera decir lo contrario, no aplica acá.
>
> **Nota de idioma:** la prosa va en español; los bloques de código mantienen sus comentarios
> en inglés.

# Plan — `webtyp/weightsc`, corrección puntual sobre la v0.1.0 ya mergeada

## Leé esta sección primero

Este repo **ya tiene el conversor completo e implementado**, mergeado como v0.1.0: lee
`safetensors` + `config.json` + `tokenizer.json`, cuantiza a int8 por fila, escribe el
artifact `.wtypw` y el `.merges`. Una revisión encontró **5 defectos puntuales**, cada uno
en un archivo distinto, ninguno estructural. Este plan corrige exactamente esos 5 puntos.

**No reescribas ningún archivo. No cambies el diseño, el CLI, el formato del artifact ni la
lógica de cuantización.** Si una parte del código no está listada abajo, no la toques —
incluidos los `map[string]TensorInfo` y `map[string]int` que ya existen en `safetensors.go` y
`convert.go`: son legítimos en este repo (ver nota de arriba) y **no** hay que convertirlos a
slice. Una versión anterior de este plan tenía una línea de checklist que sugería lo
contrario por error; la sección "Checklist de aceptación" de abajo la corrige.

## Fix 1 — `cmd/weightsc/main.go`: `printUsage` no imprime los flags porque corre antes de registrarlos

Hoy:

```go
func main() {
	if len(os.Args) <= 1 {
		printUsage()
		os.Exit(0)
	}

	inDir := flag.String("in", "", "input directory containing model.safetensors, config.json, and tokenizer.json")
	outFile := flag.String("out", "", "output .wtypw artifact file path")
	mergesOutFile := flag.String("merges-out", "", "output .merges companion file path")
	artifactID := flag.String("id", "", "artifact ID")
	version := flag.Uint("version", 0, "artifact version number")
	...
```

`printUsage` llama a `flag.PrintDefaults()`, pero en la rama sin argumentos ese llamado
ocurre **antes** de que `flag.String`/`flag.Uint` registren nada — el set de flags está
vacío y `PrintDefaults()` no imprime nada por flag.

**Arreglo:** mové las cinco declaraciones de flag (`inDir`, `outFile`, `mergesOutFile`,
`artifactID`, `version`) y la asignación de `flag.Usage` **antes** del chequeo
`len(os.Args) <= 1`. El resto de `main` (el `flag.Parse()` y todo lo que sigue) no cambia de
lugar, solo las declaraciones suben.

## Fix 2 — `cmd/weightsc/main.go`: `-version` se trunca en silencio si excede `uint32`

Hoy:

```go
version := flag.Uint("version", 0, "artifact version number")
...
artifactBytes, mergesBytes, err := weightsc.Convert(*inDir, *artifactID, uint32(*version))
```

`flag.Uint` devuelve un `*uint` (64 bits en esta plataforma). `uint32(*version)` en la línea
del `Convert` recorta en silencio cualquier valor que no entre en 32 bits — por ejemplo
`-version 4294967297` (2^32+1) pasa el chequeo `*version == 0` y se convierte en `1` sin
error ni warning.

**Arreglo:** agregá un chequeo explícito de rango inmediatamente después del bloque de
validación existente (el que ya chequea `*inDir == ""`, etc.), antes de llamar a
`weightsc.Convert`:

```go
if *version > math.MaxUint32 {
	fmt.Fprintf(os.Stderr, "Error: -version %d exceeds uint32 range (max %d)\n", *version, uint32(math.MaxUint32))
	os.Exit(1)
}
```

Agregá `"math"` al bloque de imports de `cmd/weightsc/main.go`.

## Fix 3 — `convert.go`: `parseVocab` deja tokens vacíos en silencio si el vocabulario tiene huecos

Hoy:

```go
func parseVocab(vocabMap map[string]int, vocabSize int) []string {
	maxID := -1
	for _, id := range vocabMap {
		if id > maxID {
			maxID = id
		}
	}
	size := vocabSize
	if size < maxID+1 {
		size = maxID + 1
	}

	vocab := make([]string, size)
	for tok, id := range vocabMap {
		if id >= 0 && id < size {
			vocab[id] = tok
		}
	}
	return vocab
}
```

Si `tokenizer.json` tuviera un id dentro de `[0, size)` sin token asociado (un hueco), esa
posición del slice queda como `""` sin ningún error — el artifact final embarca un token en
blanco en ese id, y el defecto es silencioso.

**Arreglo:** cambiá la firma para devolver un error y verificá que no quede ningún hueco
antes de retornar:

```go
func parseVocab(vocabMap map[string]int, vocabSize int) ([]string, error) {
	maxID := -1
	for _, id := range vocabMap {
		if id > maxID {
			maxID = id
		}
	}
	size := vocabSize
	if size < maxID+1 {
		size = maxID + 1
	}

	vocab := make([]string, size)
	filled := make([]bool, size)
	for tok, id := range vocabMap {
		if id >= 0 && id < size {
			vocab[id] = tok
			filled[id] = true
		}
	}

	for id, ok := range filled {
		if !ok {
			return nil, fmt.Errorf("vocab has no token for id %d (vocab size %d)", id, size)
		}
	}

	return vocab, nil
}
```

Actualizá el único caller, en `Convert` (`convert.go`):

```go
vocab, err := parseVocab(tokData.Model.Vocab, cfg.VocabSize)
if err != nil {
	return nil, nil, fmt.Errorf("parsing vocab from tokenizer.json: %w", err)
}
```

(hoy la línea es `vocab := parseVocab(tokData.Model.Vocab, cfg.VocabSize)`, sin chequeo de
error — reemplazala por las dos líneas de arriba, en el mismo lugar donde está hoy, antes de
la llamada a `parseMerges`).

**Test nuevo**, agregalo a `convert_test.go` junto a los tests existentes de `parseVocab` si
los hay, o como test nuevo si no los hay:

```go
func TestParseVocab_GapReturnsError(t *testing.T) {
	vocabMap := map[string]int{"a": 0, "c": 2} // hueco en id 1
	_, err := parseVocab(vocabMap, 3)
	if err == nil {
		t.Fatal("expected error for vocab gap, got nil")
	}
}

func TestParseVocab_DenseOK(t *testing.T) {
	vocabMap := map[string]int{"a": 0, "b": 1, "c": 2}
	vocab, err := parseVocab(vocabMap, 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := []string{"a", "b", "c"}
	for i, w := range want {
		if vocab[i] != w {
			t.Errorf("vocab[%d] = %q, want %q", i, vocab[i], w)
		}
	}
}
```

## Fix 4 — `go.mod`: `webtyp.com/weights` está marcado `// indirect` pero se importa directo

`convert.go` tiene `import "webtyp.com/weights"` — un import directo. Hoy `go.mod` lo lista
como:

```
webtyp.com/weights v0.1.0 // indirect
```

**Arreglo:** quitale el comentario `// indirect` a esa única línea. Las otras cinco
(`context`, `fetch`, `fmt`, `model`, `storage`) son dependencias transitivas de `weights` y
se quedan tal cual, con `// indirect`.

## Fix 5 — `convert_test.go`: el caso "neg zero" de `TestBF16ToF32_KnownValues` no puede fallar nunca

Hoy:

```go
{"neg zero", 0x8000, -0.0},
...
got := bf16ToF32(tt.bits)
if got != tt.expected && !(math.IsNaN(...) && math.IsNaN(...)) {
	t.Errorf(...)
}
```

En Go, el literal `-0.0` como constante es cero exacto (no hay "menos cero" en la aritmética
de constantes), y la comparación `!=` de punto flotante trata `+0.0` y `-0.0` como iguales
(regla IEEE 754). Si `bf16ToF32(0x8000)` alguna vez devolviera `+0.0` en vez de `-0.0` por una
regresión, este test seguiría pasando sin detectarlo.

**Arreglo:** cambiá la comparación de todo el subtest a bit-exacta, que sí distingue signo de
cero y no rompe ningún caso existente de la tabla (ninguno es NaN):

```go
got := bf16ToF32(tt.bits)
if math.Float32bits(got) != math.Float32bits(tt.expected) {
	t.Errorf("bf16ToF32(0x%04X) = %v (bits %#x), want %v (bits %#x)",
		tt.bits, got, math.Float32bits(got), tt.expected, math.Float32bits(tt.expected))
}
```

Borrá la rama `math.IsNaN` — ya no hace falta (`Float32bits` compara NaNs por bit-pattern
exacto, que es más estricto y sigue siendo correcto para esta tabla, donde no hay NaN).

Y para que `{"neg zero", 0x8000, -0.0}` compare contra el bit pattern correcto de cero
negativo, construí `tt.expected` con signo explícito en vez del literal `-0.0` (que Go
normaliza a cero positivo antes de que este test lo vea):

```go
{"neg zero", 0x8000, float32(math.Copysign(0, -1))},
```

## Checklist de aceptación

```bash
go vet ./...
gotest
```

**No** hay chequeo de `map[` acá — ver la nota de arriba y `AGENTS.md`: los maps de
`safetensors.go` y `convert.go` son código correcto tal como está, este repo no compila a
WASM y la regla de "sin `map[K]V`" no le aplica.

No hace falta `GOOS=js GOARCH=wasm` ni `tinygo` — este repo no compila a WASM.
