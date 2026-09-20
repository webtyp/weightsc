---
PLAN: "feat: webtyp/weightsc — safetensors to WTYPW1 converter for granite-embedding-97m-multilingual-r2"
TAG: v0.1.0
EXECUTOR: jules
REVIEWER: none
REPO: webtyp/weightsc
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.
> Índice maestro: https://github.com/webtyp/agent/blob/main/docs/MASTER_PLAN.md —
> §5 nota (g). El formato que se escribe está documentado en
> [`webtyp/weights`](https://github.com/webtyp/weights/blob/main/docs/PLAN.md).
>
> **Este repositorio es una herramienta host-only**, no compila a WASM. `os`, `flag`,
> `net/http` y el resto de la stdlib de Go son legítimos acá — **no los "corrijas"** aunque
> hayas visto la regla contraria en otros repos del ecosistema (`weights`, `tokenizer`,
> `transformer`). La regla de "sin stdlib" es para código que compila a WASM bajo TinyGo;
> este repo nunca lo hace. Ver `AGENTS.md` en este repo, que fija esto explícitamente.
>
> **Nota de idioma:** la prosa va en español; los bloques de código mantienen sus
> comentarios en inglés.

# Plan — `webtyp/weightsc`

## Responsabilidad única

Lee un modelo HuggingFace en formato `safetensors` (más su `config.json` y `tokenizer.json`)
y escribe el artifact binario `WTYPW1` que `webtyp/weights` sabe leer. CLI de un solo uso,
no una librería de propósito general: existe para producir el artifact real de
`granite-embedding-97m-multilingual-r2` y cualquier candidato futuro con la misma forma
(ModernBERT), no para ser un conversor universal de formatos de modelo.

**No** decide cuantización, ni formato de artifact, ni el contrato de `TokenizerConfig` — esos
ya están fijados por `webtyp/weights` (publicado, v0.1.0). Este repo los consume, no los
diseña.

## Design gate

**1. Prior art.** `llama.cpp`'s `convert_hf_to_gguf.py` (Python, lee safetensors + config.json,
escribe GGUF); `optimum`'s exportador ONNX de HuggingFace (Python, similar contrato); `ormc`
en este mismo ecosistema (Go, lee un modelo declarativo, escribe código/artifact, vive en su
propio repo con sufijo `c`). Este plan sigue el patrón de `ormc`: convertir es trabajo de
build, vive separado del runtime, y en Go porque ya existe `weights.WriteArtifact` en Go — no
hace falta un segundo lenguaje para un solo paso de conversión.

**2. Novice-name test.** `weightsc convert -in <dir> -out <file>` — un desarrollador que
conoce `ormc`/`ddlc`/`sitec` lee el nombre y ya sabe qué hace sin documentación. `convert` es
el verbo que ya usa este dominio (`convert_hf_to_gguf.py`), no se inventa uno nuevo.

**3. Complexity ledger.**
```
Conceptos nuevos para el desarrollador   +1 (un CLI de conversión, ya conocido por el patrón *c)
Repos a tocar para producir un artifact  0 (weightsc solo; no toca weights ni tokenizer)
Formas de producir un artifact WTYPW1    1 (este CLI llama a weights.WriteArtifact; nadie
                                             más construye el binario a mano)
```

**4. Dónde vive.** Repo aparte de `weights`, mismo patrón que `ormc`/`ddlc`/`sitec` (§5 nota
(g) del índice maestro). Importa `webtyp.com/weights` para el writer; no al revés.

**5. Qué borra.** Nada — es capacidad nueva. Cierra la nota (g) del índice maestro, que deja
de estar "sin plan".

## Qué convierte, exactamente

Un solo modelo por ahora: **`ibm-granite/granite-embedding-97m-multilingual-r2`**
(HuggingFace, Apache 2.0, sin autenticación). Arquitectura `ModernBertModel`. Verificado
directo desde el repo real, no de memoria — estos números son la fuente de verdad:

```
config.json (campos relevantes):
  hidden_size: 384          num_hidden_layers: 12      num_attention_heads: 12
  intermediate_size: 1536   vocab_size: 180000          layer_norm_eps: 1e-5
  attention_bias: false     mlp_bias: false             norm_bias: false
  dtype: bfloat16

model.safetensors: archivo único, 194 889 568 bytes, 74 tensores.
```

**Nombres de tensor** (verificados leyendo el header real del `.safetensors`, no inferidos):

```
embeddings.tok_embeddings.weight   [180000, 384]  BF16   ← tabla de embeddings
embeddings.norm.weight             [384]          BF16   ← LayerNorm tras el embedding
final_norm.weight                  [384]          BF16   ← LayerNorm final, antes del pooling

por cada capa i en 0..11:
  layers.{i}.attn.Wqkv.weight      [1152, 384]    BF16   ← QKV fusionado (3×384)
  layers.{i}.attn.Wo.weight        [384, 384]     BF16   ← proyección de salida de atención
  layers.{i}.attn_norm.weight      [384]          BF16   ← AUSENTE en la capa 0 (ver nota)
  layers.{i}.mlp.Wi.weight         [3072, 384]    BF16   ← gate+up fusionado (2×1536)
  layers.{i}.mlp.Wo.weight         [384, 1536]    BF16   ← proyección de bajada del MLP
  layers.{i}.mlp_norm.weight       [384]          BF16
```

**Nota — capa 0 no tiene `attn_norm`.** ModernBERT usa `Identity` ahí porque
`embeddings.norm` ya normalizó justo antes; es el diseño real del modelo, no un tensor
faltante por error. El conteo cierra: 3 + (12×6 − 1) = 74, que es exactamente el total del
header. Si tu código para 71 o 75, el error está en cómo mapeaste la capa 0.

Total: 3 + 11×6 + 5 = 74 tensores. ✓ coincide con el header real.

## Formato de origen: `safetensors`

```
[8 bytes]  N, uint64 little-endian — longitud del header JSON
[N bytes]  header JSON: {"tensor.name": {"dtype": "BF16", "shape": [...], "data_offsets": [start, end]}, ...}
[resto]    datos crudos de todos los tensores, contiguos, en el orden del header;
           data_offsets es RELATIVO al final del header (no al inicio del archivo)
```

Sin dependencias externas: `encoding/json` para el header (este repo es host-only, ver nota
de arriba) y lectura de bytes crudos con `encoding/binary`. El campo `__metadata__` del header
no es un tensor — filtralo antes de iterar.

**Conversión BF16 → float32:** un `bfloat16` son los 16 bits altos de un `float32` IEEE 754.
Conversión exacta, sin pérdida adicional:

```go
func bf16ToF32(bits uint16) float32 {
	return math.Float32frombits(uint32(bits) << 16)
}
```

## Cuantización: qué tensor se cuantiza y cómo

Regla, ya fijada por `webtyp/weights` D5 y `MASTER_PLAN.md`: **todo tensor 2D se cuantiza a
int8 con una escala por fila; todo tensor 1D (los `*_norm.weight`) se queda en float32, sin
escalas.** Cuantizar un vector de escala de LayerNorm de 384 elementos no ahorra memoria que
importe y sí mete error justo donde el modelo es más sensible — no lo hagas aunque D5 diga
"todo el artifact en int8": esa frase habla del cuerpo del transformer (las matrices), no de
los normalizadores.

Por fila, `row` es el eje 0 del shape original (antes de cualquier transposición — ver nota
de `Wqkv`/`mlp.Wi` abajo):

```go
// QuantizeRowInt8 converts one row of float32 values to int8 with a per-row scale.
// scale = max(abs(row)) / 127; q[i] = round(row[i] / scale), clamped to [-127, 127].
// A row of all zeros gets scale = 1 (avoid divide by zero) and stays all zeros.
func QuantizeRowInt8(row []float32) (q []int8, scale float32)
```

**Los tensores fusionados (`Wqkv`, `mlp.Wi`) se cuantizan como están, fila por fila, sin
separarlos.** Separar QKV en Q/K/V o gate/up acá sería una decisión de layout que le
corresponde a `webtyp/transformer` (quien sabe cómo los usa), no a este conversor. Este repo
copia forma y datos; `transformer` decide cómo cortarlos en tiempo de carga.

## Vocabulario y merges: **no van los dos en el artifact**

`weights.TokenizerConfig` (ya publicado, v0.1.0) tiene `Vocab []string` pero **no** un campo
para las reglas de merge de BPE — el diseño original asumía WordPiece, que no necesita
merges. Añadir un campo a un tipo público ya publicado es un cambio de API en un repo ajeno,
fuera del alcance de este plan (ver Design gate §4: este repo consume el contrato de
`weights`, no lo cambia).

**Decisión, para no dejarlo abierto:** este CLI escribe **dos archivos**:

1. El artifact `.wtypw` de siempre, vía `weights.WriteArtifact(id, version, tok, inputs)`,
   con `tok.Vocab` poblado (orden = id de token, ver abajo) y `tok.Lowercase`/`tok.StripAccents`
   **ambos en `false`** — el tokenizer real de este modelo no hace ninguna de las dos cosas
   (ver `webtyp/tokenizer/docs/PLAN.md`, que es el otro plan de esta tanda).
2. Un archivo plano `.merges` — una línea por regla, en el mismo orden que aparecen en
   `tokenizer.json`, formato `"tok1 tok2"` (los mismos dos campos que trae el JSON, separados
   por un espacio, sin comillas). El rank de una regla es su número de línea (0-indexed).
   `webtyp/tokenizer` lo consume como `[]string` — cero parseo de JSON de su lado.

Ambos se generan de la misma corrida, para el mismo modelo, y viajan juntos: la aplicación que
los consuma (fuera de este plan) es responsable de no mezclar el `.wtypw` de una versión con
el `.merges` de otra. Ese emparejamiento es trabajo de `webtyp/embed` cuando se escriba su
adaptador — no de este repo.

**Vocabulario, orden y contenido:** `tokenizer.json` → `model.vocab` es un objeto
`{"token string": id}`. Invertilo a un `[]string` indexado por id (`vocab[id] = "token
string"`), del 0 al `vocab_size-1` (180000, verificado en `config.json`). Los tokens son
**strings byte-level de GPT-2** (cada byte crudo mapeado a un carácter Unicode imprimible,
alfabeto de 256 símbolos — ver la sección de `webtyp/tokenizer/docs/PLAN.md` sobre esto): se
copian tal cual, no se decodifican a UTF-8 acá. Decodificarlos es trabajo de `tokenizer`, no
de este conversor.

**`merges`:** `tokenizer.json` → `model.merges` es un array. Verificá en el archivo real si
cada entrada es `["tok1", "tok2"]` (par) o `"tok1 tok2"` (string) — las dos formas existen
según la versión de `tokenizers` que exportó el archivo; normalizá a `"tok1 tok2"` al escribir
el `.merges`, sea cual sea el formato de origen.

## CLI

```bash
weightsc -in <dir> -out <file.wtypw> -merges-out <file.merges> -id <artifact-id> -version <uint32>
```

`-in` es un directorio que contiene `model.safetensors`, `config.json` y `tokenizer.json` — no
los descarga; asumí que ya están ahí (`huggingface-cli download` o `curl`, documentado en el
`README.md` de este repo, no en el código). Sin argumentos, imprime el uso a stdout y sale con
`0` (contrato de ejecución de `core-principles`, aplica igual en un CLI host-only).

Estructura, por la regla de `cmd/` delgado (`plan-authoring`):

```
convert.go     // toda la lógica: parseo de safetensors, cuantización, escritura — testeable
cmd/weightsc/main.go   // solo flag parsing + llamar a convert.Run(...) + os.Exit
```

## Tests

Sin descargar el modelo real de 195 MB en CI: un fixture sintético de 2-3 tensores pequeños
(un 2D y un 1D) con el mismo layout de nombres (`embeddings.norm.weight`,
`embeddings.tok_embeddings.weight` achicado a, digamos, 8 filas), escrito a mano como
safetensors válido en el test.

| Test | Verifica |
|---|---|
| `TestBF16ToF32_KnownValues` | un puñado de bits BF16 conocidos, incluido 0, negativo, y el valor más cercano a 1.0 |
| `TestQuantizeRowInt8_RoundTrip` | cuantizar y dequantizar (`q[i]*scale`) queda dentro de `scale/2` del original |
| `TestQuantizeRowInt8_AllZeros` | scale = 1, sin división por cero |
| `TestParseSafetensorsHeader_Fixture` | el fixture sintético produce los tensores esperados, con `__metadata__` filtrado |
| `TestConvert_RoundTripThroughWeights` | el artifact escrito se abre con `weights.Open` sin error, y sus tensores int8 dequantizados están dentro de tolerancia de los float32 de entrada |
| `TestConvert_MergesFileFormat` | el `.merges` tiene una línea por regla, en el orden de entrada |

## Checklist de aceptación

```bash
go vet ./...
gotest
grep -rn "map\[" --include="*.go" . | grep -v _test.go   # → vacío (sigue aplicando: aunque
                                                           # este repo no compila a wasm, es
                                                           # buena práctica y evita sorpresas
                                                           # si algo de acá se reusa después)
```

No hace falta `GOOS=js GOARCH=wasm` ni `tinygo` acá — este repo no compila a WASM (ver nota de
arriba y `AGENTS.md`).
