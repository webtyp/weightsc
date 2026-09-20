# Tipos numéricos en WASM, y por qué convertimos BF16 e int8

Para un dev junior. Si ya sabés qué son `i32`/`f32` en WASM y qué es la cuantización, este
documento no te dice nada nuevo.

## La idea equivocada, primero

"WASM solo tiene enteros de 32 y 64 bits" — **no**. La especificación de WebAssembly define
**cuatro** tipos de valor desde el día uno (el "MVP", 2017):

| Tipo | Qué es |
|---|---|
| `i32` | entero de 32 bits |
| `i64` | entero de 64 bits |
| `f32` | **float de 32 bits, IEEE 754 — nativo** |
| `f64` | **float de 64 bits, IEEE 754 — nativo** |

`f32` y `f64` no son un truco montado sobre enteros. Tienen sus propias instrucciones
(`f32.add`, `f32.mul`, `f32.sqrt`, `f32.load`, `f32.store`, ...) y el motor WASM del navegador
las compila directo a las instrucciones de punto flotante reales del procesador (SSE en x86,
NEON en ARM) — el mismo hardware que usaría un programa en C nativo. No hay emulación, no hay
penalidad por "no ser un entero".

De dónde viene la confusión, probablemente: en JavaScript **todos** los números son
`float64` por debajo (no existe un tipo entero real en JS, salvo `BigInt`), así que quien
viene de JS a WASM a veces da vuelta la idea: "como JS no tiene enteros reales, WASM debe
tener solo enteros". Es al revés — WASM tiene los dos, y es JS el que solo tiene un tipo
numérico por defecto.

Lo único donde `i32` manda es en el **direccionamiento de memoria**: un puntero/offset dentro
de la memoria lineal de WASM es un `i32` (o `i64` en la variante wasm64, que casi nadie usa
todavía). Pero lo que hay *en* esa dirección de memoria puede ser cualquier cosa — bytes
crudos que herramientas como `f32.load`/`f32.store` interpretan como floats. `vector.Dot`,
`MatmulT`, todo el código de `transformer` — opera en `f32` de punta a punta, y eso corre
sobre hardware de punto flotante real, no sobre una simulación con enteros.

**Entonces `weights`/`transformer` ya están en el tipo correcto.** El `[]float32` de Go
compila a `f32` de WASM 1:1. Ahí no hay conversión que evitar ni que optimizar — es el tipo
nativo del problema.

## Entonces, ¿por qué `weightsc` convierte BF16 a `float32`?

Porque **BF16 no es un tipo que WASM (ni Go, ni casi ningún hardware de propósito general)
sepa operar directamente.** Es un formato de *almacenamiento*, no un tipo aritmético.

`BF16` ("brain float 16", inventado en Google Brain para entrenar redes) son los **16 bits
altos** de un `float32` IEEE 754 — mismo exponente de 8 bits que `float32` (por eso no pierde
rango), pero solo 7 bits de mantisa en vez de 23 (por eso pierde precisión). Así quedó
guardado el modelo original en PyTorch — `config.json` de `granite-embedding-97m-multilingual-r2`
lo dice explícito: `"dtype": "bfloat16"`.

Ninguna ALU de propósito general —ni la de tu CPU, ni la que expone WASM— suma o multiplica
`bf16` directo. Para operar sobre esos números hace falta primero llevarlos a un tipo que el
hardware sí entiende: `float32`. La conversión es exacta y barata, **una sola línea, sin
pérdida adicional** a la que ya tiene el propio BF16:

```go
func bf16ToF32(bits uint16) float32 {
	return math.Float32frombits(uint32(bits) << 16)
}
```

(los 16 bits bajos de la mantisa quedan en cero — no se inventa precisión que no estaba, pero
tampoco se pierde nada que BF16 ya tuviera).

**Esto pasa una sola vez, offline, en `weightsc` — nunca en el navegador.** El artifact
`WTYPW1` que se descarga y cachea **no contiene BF16 en ningún lado**: `weightsc` ya lo
convirtió antes de escribir el archivo. El navegador jamás ve un valor BF16.

## ¿Y el int8 en el artifact? ¿Eso también hay que "destransformarlo"?

Sí, pero es un problema distinto y con una razón distinta: no es "int8 no es un tipo
aritmético válido" (si lo es, `i32` puede sumarlo/restarlo sin problema) — es que **el int8
del artifact no representa el peso real, representa el peso *escalado*.**

`weightsc` cuantiza cada fila de una matriz de pesos así: busca el valor de mayor magnitud
de la fila, lo divide entre 127, y usa eso como `scale`; cada peso de esa fila se guarda como
`round(peso / scale)`, un entero de -127 a 127. Eso **no es el peso** — es una aproximación
entera de "cuántas escalas hay que sumar para llegar al peso real". Para recuperar el valor
utilizable hace falta la operación inversa:

```go
peso_real ≈ int8_guardado * scale   // dequantización — un float32 por valor
```

La razón de guardar así **no es** de tipos de WASM — es de **tamaño de descarga**: un `int8`
por peso pesa un cuarto que un `float32` por peso. Un artifact de ~100 MB en `float32` baja a
~25 MB en `int8` — la diferencia entre esperar y no esperar la primera carga en el navegador.
El costo es que hay que dequantizar (`int8 * scale`) antes de poder usar ese valor en un
`vector.Dot` — WASM no tiene una instrucción de "producto punto sobre enteros escalados", así
que ese paso es inevitable en algún punto de la cadena.

**Dónde pasa esa dequantización, y por qué no es trabajo de más:**

- El **cuerpo del transformer** (~28,3M parámetros, las matrices de atención y MLP) se
  dequantiza **una vez, al cargar el artifact**, y se queda en `float32` en memoria mientras
  dura la sesión del navegador — se usa en cada consulta, así que no tiene sentido
  redequantizar cada vez.
- La **tabla de embeddings** (180 000 filas × 384 — sería ~276 MB en `float32`) se queda en
  `int8` en memoria, y **solo** se dequantiza la fila de cada token que aparece en la consulta
  real — unas 20-30 filas de 180 000. Dequantizar la tabla entera para usar el 0,01% sería
  exactamente el trabajo innecesario que esto evita.

## Resumen para no perderse

| Conversión | Dónde ocurre | Cuántas veces | Por qué hace falta |
|---|---|---|---|
| BF16 → float32 | `weightsc`, offline | una vez, al construir el artifact | BF16 no es un tipo aritmético de WASM/CPU; float32 sí |
| int8 → float32 (cuerpo) | navegador, al cargar | una vez por sesión | ahorra 4× de descarga; el cómputo real necesita float32 |
| int8 → float32 (embeddings) | navegador, por consulta | solo las filas de los tokens usados | evita destransformar 276 MB para usar 20 filas |

Ninguna de las tres es una conversión de más: cada una existe porque resuelve un problema real
(BF16 no es operable, el tamaño de descarga importa, decodificar la tabla entera sería
desperdiciar memoria) y cada una pasa **el mínimo número de veces posible** — nunca se
reconvierte algo que ya está en el tipo que hace falta.
