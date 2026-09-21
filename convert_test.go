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

func TestBF16ToF32_KnownValues(t *testing.T) {
	tests := []struct {
		name     string
		bits     uint16
		expected float32
	}{
		{"zero", 0x0000, 0.0},
		{"neg zero", 0x8000, -0.0},
		{"one", 0x3F80, 1.0},
		{"neg one", 0xBF80, -1.0},
		{"half", 0x3F00, 0.5},
		{"two", 0x4000, 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := bf16ToF32(tt.bits)
			if got != tt.expected && !(math.IsNaN(float64(got)) && math.IsNaN(float64(tt.expected))) {
				t.Errorf("bf16ToF32(0x%04X) = %v, want %v", tt.bits, got, tt.expected)
			}
		})
	}
}

func TestQuantizeRowInt8_RoundTrip(t *testing.T) {
	row := []float32{0.0, 0.5, -0.75, 1.25, -2.0, 1.99}
	q, scale := QuantizeRowInt8(row)

	if len(q) != len(row) {
		t.Fatalf("len(q) = %d, want %d", len(q), len(row))
	}
	if scale <= 0 {
		t.Fatalf("scale = %v, want > 0", scale)
	}

	for i := range row {
		dequant := float32(q[i]) * scale
		diff := float64(math.Abs(float64(dequant - row[i])))
		tol := float64(scale/2.0) + 1e-5
		if diff > tol {
			t.Errorf("row[%d] = %v, q = %d, scale = %v, dequant = %v, diff = %v > tol %v",
				i, row[i], q[i], scale, dequant, diff, tol)
		}
	}
}

func TestQuantizeRowInt8_AllZeros(t *testing.T) {
	row := []float32{0.0, 0.0, 0.0, 0.0}
	q, scale := QuantizeRowInt8(row)

	if scale != 1.0 {
		t.Errorf("scale = %v, want 1.0", scale)
	}
	for i, v := range q {
		if v != 0 {
			t.Errorf("q[%d] = %d, want 0", i, v)
		}
	}
}

func createSyntheticSafetensors(t *testing.T, dir string) string {
	t.Helper()
	// Create float32 values and convert to BF16
	f1D := []float32{1.0, 2.0, 3.0, 4.0}                           // 4 elements -> 8 bytes
	f2D := []float32{0.1, 0.2, 0.3, 0.4, -0.5, -0.6, -0.7, -0.8} // 8 elements -> 16 bytes

	var payload bytes.Buffer
	for _, v := range f1D {
		bits := math.Float32bits(v)
		bf16 := uint16(bits >> 16)
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], bf16)
		payload.Write(b[:])
	}
	for _, v := range f2D {
		bits := math.Float32bits(v)
		bf16 := uint16(bits >> 16)
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], bf16)
		payload.Write(b[:])
	}

	headerMap := map[string]any{
		"__metadata__": map[string]string{"format": "pt"},
		"embeddings.norm.weight": map[string]any{
			"dtype":        "BF16",
			"shape":        []int{4},
			"data_offsets": []int{0, 8},
		},
		"embeddings.tok_embeddings.weight": map[string]any{
			"dtype":        "BF16",
			"shape":        []int{2, 4},
			"data_offsets": []int{8, 24},
		},
	}

	headerJSON, err := json.Marshal(headerMap)
	if err != nil {
		t.Fatalf("marshaling synthetic header: %v", err)
	}

	stPath := filepath.Join(dir, "model.safetensors")
	f, err := os.Create(stPath)
	if err != nil {
		t.Fatalf("creating synthetic safetensors: %v", err)
	}
	defer f.Close()

	var hLenBytes [8]byte
	binary.LittleEndian.PutUint64(hLenBytes[:], uint64(len(headerJSON)))
	f.Write(hLenBytes[:])
	f.Write(headerJSON)
	f.Write(payload.Bytes())

	return stPath
}

func TestParseSafetensorsHeader_Fixture(t *testing.T) {
	dir := t.TempDir()
	stPath := createSyntheticSafetensors(t, dir)

	st, err := OpenSafetensors(stPath)
	if err != nil {
		t.Fatalf("OpenSafetensors failed: %v", err)
	}
	defer st.Close()

	if _, found := st.Tensors["__metadata__"]; found {
		t.Errorf("__metadata__ was not filtered out from Tensors map")
	}

	norm, found := st.Tensors["embeddings.norm.weight"]
	if !found {
		t.Fatalf("embeddings.norm.weight not found")
	}
	if norm.DType != "BF16" || len(norm.Shape) != 1 || norm.Shape[0] != 4 {
		t.Errorf("unexpected norm info: %+v", norm)
	}

	tok, found := st.Tensors["embeddings.tok_embeddings.weight"]
	if !found {
		t.Fatalf("embeddings.tok_embeddings.weight not found")
	}
	if tok.DType != "BF16" || len(tok.Shape) != 2 || tok.Shape[0] != 2 || tok.Shape[1] != 4 {
		t.Errorf("unexpected tok info: %+v", tok)
	}
}

func TestConvert_RoundTripThroughWeights(t *testing.T) {
	dir := t.TempDir()
	createSyntheticSafetensors(t, dir)

	cfgContent := `{"vocab_size": 4}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfgContent), 0644); err != nil {
		t.Fatalf("writing config.json: %v", err)
	}

	tokContent := `{
		"model": {
			"vocab": {"a": 0, "b": 1, "c": 2, "d": 3},
			"merges": ["a b", ["c", "d"]]
		}
	}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokContent), 0644); err != nil {
		t.Fatalf("writing tokenizer.json: %v", err)
	}

	artBytes, mergesBytes, err := Convert(dir, "test-model-id", 1)
	if err != nil {
		t.Fatalf("Convert failed: %v", err)
	}

	art, err := weights.Open(artBytes)
	if err != nil {
		t.Fatalf("weights.Open failed: %v", err)
	}

	if art.ID != "test-model-id" {
		t.Errorf("art.ID = %q, want %q", art.ID, "test-model-id")
	}
	if art.Version != 1 {
		t.Errorf("art.Version = %d, want 1", art.Version)
	}
	if art.Tokenizer.Lowercase || art.Tokenizer.StripAccents {
		t.Errorf("tokenizer config options should be false, got Lowercase=%v StripAccents=%v",
			art.Tokenizer.Lowercase, art.Tokenizer.StripAccents)
	}
	if len(art.Tokenizer.Vocab) != 4 || art.Tokenizer.Vocab[0] != "a" || art.Tokenizer.Vocab[3] != "d" {
		t.Errorf("unexpected vocab: %v", art.Tokenizer.Vocab)
	}

	// Verify 1D tensor (Float32)
	normTensor, found := art.Tensor("embeddings.norm.weight")
	if !found {
		t.Fatalf("embeddings.norm.weight tensor missing")
	}
	if normTensor.DType != weights.Float32 {
		t.Errorf("embeddings.norm.weight dtype = %v, want Float32", normTensor.DType)
	}
	f32s, err := normTensor.Float32s()
	if err != nil {
		t.Fatalf("Float32s error: %v", err)
	}
	if len(f32s) != 4 {
		t.Fatalf("len(f32s) = %d, want 4", len(f32s))
	}

	// Verify 2D tensor (Int8 with row scales)
	tokTensor, found := art.Tensor("embeddings.tok_embeddings.weight")
	if !found {
		t.Fatalf("embeddings.tok_embeddings.weight tensor missing")
	}
	if tokTensor.DType != weights.Int8 {
		t.Errorf("embeddings.tok_embeddings.weight dtype = %v, want Int8", tokTensor.DType)
	}
	if len(tokTensor.Scales) != 2 {
		t.Errorf("tokTensor.Scales count = %d, want 2", len(tokTensor.Scales))
	}

	// Verify merges file
	expectedMerges := "a b\nc d\n"
	if string(mergesBytes) != expectedMerges {
		t.Errorf("mergesBytes = %q, want %q", string(mergesBytes), expectedMerges)
	}
}

func TestConvert_MergesFileFormat(t *testing.T) {
	dir := t.TempDir()
	createSyntheticSafetensors(t, dir)

	cfgContent := `{"vocab_size": 2}`
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfgContent), 0644)

	tokContent := `{
		"model": {
			"vocab": {"x": 0, "y": 1},
			"merges": ["x y\n", ["y", "x"]]
		}
	}`
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokContent), 0644)

	_, mergesBytes, err := Convert(dir, "test-merges", 1)
	if err != nil {
		t.Fatalf("Convert failed: %v", err)
	}

	expected := "x y\ny x\n"
	if string(mergesBytes) != expected {
		t.Errorf("mergesBytes = %q, want %q", string(mergesBytes), expected)
	}
}
