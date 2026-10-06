package weightsc

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"webtyp.com/weights"
)

// writeBF16Model writes a checkpoint with one rows×cols "weight" tensor in BF16 and returns the
// values as BF16 rounds them.
func writeBF16Model(t *testing.T, dir string, rows, cols int) []float32 {
	t.Helper()
	vals := make([]float32, rows*cols)
	var payload bytes.Buffer
	for i := range vals {
		v := float32(math.Sin(float64(i)*0.37)) * float32(1+i%5)
		bf16 := uint16(math.Float32bits(v) >> 16)
		vals[i] = math.Float32frombits(uint32(bf16) << 16)
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], bf16)
		payload.Write(b[:])
	}
	header, _ := json.Marshal(map[string]any{"weight": map[string]any{
		"dtype": "BF16", "shape": []int{rows, cols}, "data_offsets": []int{0, payload.Len()},
	}})
	var hl [8]byte
	binary.LittleEndian.PutUint64(hl[:], uint64(len(header)))
	st := append(append(hl[:], header...), payload.Bytes()...)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), st, 0644); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"vocab_size": 2}`), 0644)
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"model":{"vocab":{"a":0,"b":1},"merges":[]}}`), 0644)
	return vals
}

func convertWeight(t *testing.T, dir string, q Quant) weights.Tensor {
	t.Helper()
	b, _, err := Convert(dir, Options{ID: "m", Version: 1, Quant: q})
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}
	art, err := weights.Open(b)
	if err != nil {
		t.Fatalf("weights.Open: %v", err)
	}
	w, ok := art.Tensor("weight")
	if !ok {
		t.Fatal("tensor weight not found")
	}
	return w
}

func TestConvert_Int4Block32ThroughWeights(t *testing.T) {
	dir := t.TempDir()
	const rows, cols = 3, 64
	vals := writeBF16Model(t, dir, rows, cols)
	w := convertWeight(t, dir, QuantInt4Block32)
	if w.DType != weights.Int4Block32 || len(w.Data) != rows*cols/2 || len(w.Scales) != rows*cols/32 {
		t.Fatalf("dtype %q, %d bytes, %d scales", w.DType, len(w.Data), len(w.Scales))
	}
	dst := make([]float32, cols)
	for r := 0; r < rows; r++ {
		if err := w.DequantRow(dst, r); err != nil {
			t.Fatal(err)
		}
		for c, v := range dst {
			step := math.Abs(float64(w.Scales[r*cols/32+c/32]))
			orig := float64(vals[r*cols+c])
			// Q4_0 clips values beyond 7 steps on the side opposite the block's largest value.
			bound := math.Max(step/2, math.Abs(orig)-7*step)
			if d := math.Abs(float64(v) - orig); d > bound+1e-6 {
				t.Fatalf("row %d col %d: %v decoded as %v (step %v)", r, c, orig, v, step)
			}
		}
	}
}

func TestConvert_Int4FallsBackToInt8(t *testing.T) {
	dir := t.TempDir()
	writeBF16Model(t, dir, 2, 40)
	if w := convertWeight(t, dir, QuantInt4Block32); w.DType != weights.Int8Block32 {
		t.Errorf("40 columns: dtype %q, want Int8Block32", w.DType)
	}
}

func TestOptions_ValidateInt4(t *testing.T) {
	if err := (Options{ID: "m", Version: 1, Quant: QuantInt4Block32}).Validate(); err != nil {
		t.Errorf("int4-block32 rejected: %v", err)
	}
	err := (Options{ID: "m", Version: 1, Quant: "int2"}).Validate()
	if err == nil || !bytes.Contains([]byte(err.Error()), []byte("int4-block32")) {
		t.Errorf("unknown quant error %v does not list int4-block32", err)
	}
}
