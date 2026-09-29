package weightsc

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"webtyp.com/weights"
)

func TestParseVocab_GapIsPadded(t *testing.T) {
	vocabMap := map[string]int{"a": 0, "c": 2} // gap at id 1
	vocab, err := parseVocab(vocabMap, nil, 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := []string{"a", "<|pad_1|>", "c"}
	if len(vocab) != len(want) {
		t.Fatalf("len(vocab) = %d, want %d", len(vocab), len(want))
	}
	for i, w := range want {
		if vocab[i] != w {
			t.Errorf("vocab[%d] = %q, want %q", i, vocab[i], w)
		}
	}
}

func TestParseVocab_DenseOK(t *testing.T) {
	vocabMap := map[string]int{"a": 0, "b": 1, "c": 2}
	vocab, err := parseVocab(vocabMap, nil, 3)
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

func TestBF16ToF32_KnownValues(t *testing.T) {
	tests := []struct {
		name     string
		bits     uint16
		expected float32
	}{
		{"zero", 0x0000, 0.0},
		{"neg zero", 0x8000, float32(math.Copysign(0, -1))},
		{"one", 0x3F80, 1.0},
		{"neg one", 0xBF80, -1.0},
		{"half", 0x3F00, 0.5},
		{"two", 0x4000, 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := bf16ToF32(tt.bits)
			if math.Float32bits(got) != math.Float32bits(tt.expected) {
				t.Errorf("bf16ToF32(0x%04X) = %v (bits %#x), want %v (bits %#x)",
					tt.bits, got, math.Float32bits(got), tt.expected, math.Float32bits(tt.expected))
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
	f1D := []float32{1.0, 2.0, 3.0, 4.0}                         // 4 elements -> 8 bytes
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

	opts := Options{
		ID:      "test-model-id",
		Version: 1,
	}

	artBytes, mergesBytes, err := Convert(dir, opts)
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

	opts := Options{
		ID:      "test-merges",
		Version: 1,
	}

	_, mergesBytes, err := Convert(dir, opts)
	if err != nil {
		t.Fatalf("Convert failed: %v", err)
	}

	expected := "x y\ny x\n"
	if string(mergesBytes) != expected {
		t.Errorf("mergesBytes = %q, want %q", string(mergesBytes), expected)
	}
}

func TestOptions_Validate(t *testing.T) {
	tests := []struct {
		name    string
		opts    Options
		wantErr string
	}{
		{
			name:    "empty ID",
			opts:    Options{ID: "", Version: 1},
			wantErr: "weightsc: Options.ID is required",
		},
		{
			name:    "zero Version",
			opts:    Options{ID: "test", Version: 0},
			wantErr: "weightsc: Options.Version must be greater than zero",
		},
		{
			name:    "unknown Quant",
			opts:    Options{ID: "test", Version: 1, Quant: "invalid"},
			wantErr: `weightsc: unknown quantization "invalid" (want int8-row, int8-block32 or float32)`,
		},
		{
			name:    "valid defaults",
			opts:    Options{ID: "test", Version: 1},
			wantErr: "",
		},
		{
			name:    "valid block32",
			opts:    Options{ID: "test", Version: 1, Quant: QuantInt8Block32},
			wantErr: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.opts.Validate()
			if tt.wantErr == "" {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			} else {
				if err == nil || err.Error() != tt.wantErr {
					t.Errorf("got error %v, want %q", err, tt.wantErr)
				}
			}
		})
	}
}

func TestConvert_DefaultIsUnchanged(t *testing.T) {
	dir := t.TempDir()
	createSyntheticSafetensors(t, dir)

	cfgContent := `{"vocab_size": 4}`
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfgContent), 0644)

	tokContent := `{
		"model": {
			"vocab": {"a": 0, "b": 1, "c": 2, "d": 3},
			"merges": ["a b", ["c", "d"]]
		}
	}`
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokContent), 0644)

	art1, merges1, err := Convert(dir, Options{ID: "test", Version: 1})
	if err != nil {
		t.Fatalf("Convert 1 failed: %v", err)
	}

	art2, merges2, err := Convert(dir, Options{ID: "test", Version: 1, Quant: QuantInt8Row})
	if err != nil {
		t.Fatalf("Convert 2 failed: %v", err)
	}

	if !bytes.Equal(art1, art2) {
		t.Errorf("artifact bytes differ between default Quant and QuantInt8Row")
	}
	if !bytes.Equal(merges1, merges2) {
		t.Errorf("merges bytes differ between default Quant and QuantInt8Row")
	}
}

func TestConvert_Int8Block32ThroughWeights(t *testing.T) {
	dir := t.TempDir()

	// 2 rows of 40 elements float32 values
	rng := rand.New(rand.NewSource(123))
	f2D := make([]float32, 80)
	for i := range f2D {
		f2D[i] = (rng.Float32()*2 - 1) * 3.0
	}

	var payload bytes.Buffer
	for _, v := range f2D {
		bits := math.Float32bits(v)
		bf16 := uint16(bits >> 16)
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], bf16)
		payload.Write(b[:])
	}

	headerMap := map[string]any{
		"weight": map[string]any{
			"dtype":        "BF16",
			"shape":        []int{2, 40},
			"data_offsets": []int{0, 160},
		},
	}
	headerJSON, _ := json.Marshal(headerMap)

	stPath := filepath.Join(dir, "model.safetensors")
	f, _ := os.Create(stPath)
	var hLenBytes [8]byte
	binary.LittleEndian.PutUint64(hLenBytes[:], uint64(len(headerJSON)))
	f.Write(hLenBytes[:])
	f.Write(headerJSON)
	f.Write(payload.Bytes())
	f.Close()

	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"vocab_size": 2}`), 0644)
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"model":{"vocab":{"a":0,"b":1},"merges":[]}}`), 0644)

	artBytes, _, err := Convert(dir, Options{ID: "block32-test", Version: 1, Quant: QuantInt8Block32})
	if err != nil {
		t.Fatalf("Convert failed: %v", err)
	}

	art, err := weights.Open(artBytes)
	if err != nil {
		t.Fatalf("weights.Open failed: %v", err)
	}

	tTensor, found := art.Tensor("weight")
	if !found {
		t.Fatalf("tensor weight not found")
	}

	if tTensor.DType != weights.Int8Block32 {
		t.Errorf("DType = %v, want Int8Block32", tTensor.DType)
	}

	if len(tTensor.Scales) != 4 {
		t.Fatalf("len(Scales) = %d, want 4", len(tTensor.Scales))
	}

	// Test DequantRow tolerance
	for r := 0; r < 2; r++ {
		row := make([]float32, 40)
		if err := tTensor.DequantRow(row, r); err != nil {
			t.Fatalf("DequantRow failed for row %d: %v", r, err)
		}
		for c := 0; c < 40; c++ {
			orig := f2D[r*40+c]
			dequant := row[c]
			diff := math.Abs(float64(dequant - orig))
			blockIdx := r*2 + c/32
			scale := tTensor.Scales[blockIdx]
			tol := float64(scale/2.0) + 1e-2 // BF16 precision + int8 quantization error
			if diff > tol {
				t.Errorf("row %d col %d: orig %v, dequant %v, diff %v > tol %v", r, c, orig, dequant, diff, tol)
			}
		}
	}
}

func TestConvert_PrefixFilters(t *testing.T) {
	dir := t.TempDir()

	var payload bytes.Buffer
	// 3 tensors, each 4 floats = 16 bytes
	f32s := []float32{1.0, 2.0, 3.0, 4.0}
	for i := 0; i < 3; i++ {
		for _, v := range f32s {
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
			payload.Write(b[:])
		}
	}

	headerMap := map[string]any{
		"model.language_model.a": map[string]any{"dtype": "F32", "shape": []int{4}, "data_offsets": []int{0, 16}},
		"model.visual.b":         map[string]any{"dtype": "F32", "shape": []int{4}, "data_offsets": []int{16, 32}},
		"mtp.c":                  map[string]any{"dtype": "F32", "shape": []int{4}, "data_offsets": []int{32, 48}},
	}
	headerJSON, _ := json.Marshal(headerMap)

	stPath := filepath.Join(dir, "model.safetensors")
	f, _ := os.Create(stPath)
	var hLenBytes [8]byte
	binary.LittleEndian.PutUint64(hLenBytes[:], uint64(len(headerJSON)))
	f.Write(hLenBytes[:])
	f.Write(headerJSON)
	f.Write(payload.Bytes())
	f.Close()

	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"vocab_size": 2}`), 0644)
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"model":{"vocab":{"a":0,"b":1},"merges":[]}}`), 0644)

	artBytes, _, err := Convert(dir, Options{
		ID:      "prefix-test",
		Version: 1,
		Prefix:  "model.language_model.",
	})
	if err != nil {
		t.Fatalf("Convert failed: %v", err)
	}

	art, err := weights.Open(artBytes)
	if err != nil {
		t.Fatalf("weights.Open failed: %v", err)
	}

	if _, found := art.Tensor("model.language_model.a"); !found {
		t.Errorf("model.language_model.a should be present")
	}
	if _, found := art.Tensor("model.visual.b"); found {
		t.Errorf("model.visual.b should be skipped")
	}
	if _, found := art.Tensor("mtp.c"); found {
		t.Errorf("mtp.c should be skipped")
	}
}

func TestConvert_ThreeDimensionalIsFloat32(t *testing.T) {
	dir := t.TempDir()

	// [3][1][4] tensor = 12 floats = 48 bytes
	f32s := make([]float32, 12)
	for i := range f32s {
		f32s[i] = float32(i + 1)
	}

	var payload bytes.Buffer
	for _, v := range f32s {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(v))
		payload.Write(b[:])
	}

	headerMap := map[string]any{
		"conv3d": map[string]any{"dtype": "F32", "shape": []int{3, 1, 4}, "data_offsets": []int{0, 48}},
	}
	headerJSON, _ := json.Marshal(headerMap)

	stPath := filepath.Join(dir, "model.safetensors")
	f, _ := os.Create(stPath)
	var hLenBytes [8]byte
	binary.LittleEndian.PutUint64(hLenBytes[:], uint64(len(headerJSON)))
	f.Write(hLenBytes[:])
	f.Write(headerJSON)
	f.Write(payload.Bytes())
	f.Close()

	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"vocab_size": 2}`), 0644)
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"model":{"vocab":{"a":0,"b":1},"merges":[]}}`), 0644)

	artBytes, _, err := Convert(dir, Options{ID: "3d-test", Version: 1})
	if err != nil {
		t.Fatalf("Convert failed: %v", err)
	}

	art, err := weights.Open(artBytes)
	if err != nil {
		t.Fatalf("weights.Open failed: %v", err)
	}

	tensor, found := art.Tensor("conv3d")
	if !found {
		t.Fatalf("conv3d tensor missing")
	}

	if tensor.DType != weights.Float32 {
		t.Errorf("DType = %v, want Float32", tensor.DType)
	}
	if len(tensor.Shape) != 3 || tensor.Shape[0] != 3 || tensor.Shape[1] != 1 || tensor.Shape[2] != 4 {
		t.Errorf("Shape = %v, want [3 1 4]", tensor.Shape)
	}

	gotF32s, err := tensor.Float32s()
	if err != nil {
		t.Fatalf("Float32s failed: %v", err)
	}
	for i, want := range f32s {
		if gotF32s[i] != want {
			t.Errorf("gotF32s[%d] = %v, want %v", i, gotF32s[i], want)
		}
	}
}

func TestConvert_TextConfigVocabAndAddedTokens(t *testing.T) {
	dir := t.TempDir()

	createSyntheticSafetensors(t, dir)

	cfgContent := `{"text_config":{"vocab_size": 8}}`
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfgContent), 0644)

	tokContent := `{
		"model": {
			"vocab": {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
			"merges": []
		},
		"added_tokens": [
			{"id": 5, "content": "<|im_start|>"}
		]
	}`
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokContent), 0644)

	artBytes, _, err := Convert(dir, Options{ID: "text-config-test", Version: 1})
	if err != nil {
		t.Fatalf("Convert failed: %v", err)
	}

	art, err := weights.Open(artBytes)
	if err != nil {
		t.Fatalf("weights.Open failed: %v", err)
	}

	vocab := art.Tokenizer.Vocab
	if len(vocab) != 8 {
		t.Fatalf("len(vocab) = %d, want 8", len(vocab))
	}

	wantVocab := []string{"a", "b", "c", "d", "e", "<|im_start|>", "<|pad_6|>", "<|pad_7|>"}
	for i, want := range wantVocab {
		if vocab[i] != want {
			t.Errorf("vocab[%d] = %q, want %q", i, vocab[i], want)
		}
	}
}

func TestConvert_AddedTokenConflict(t *testing.T) {
	dir := t.TempDir()

	createSyntheticSafetensors(t, dir)

	cfgContent := `{"vocab_size": 4}`
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfgContent), 0644)

	tokContent := `{
		"model": {
			"vocab": {"a": 0, "b": 1, "c": 2, "d": 3},
			"merges": []
		},
		"added_tokens": [
			{"id": 1, "content": "conflict"}
		]
	}`
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokContent), 0644)

	_, _, err := Convert(dir, Options{ID: "conflict-test", Version: 1})
	if err == nil {
		t.Fatal("expected conflict error, got nil")
	}

	wantSub := `weightsc: token id 1 is both "b" and "conflict"`
	if !bytes.Contains([]byte(err.Error()), []byte(wantSub)) {
		t.Errorf("err = %q, want to contain %q", err.Error(), wantSub)
	}
}

// Hugging Face publishes many checkpoints as shards plus model.safetensors.index.json
// (Qwen3.5-0.8B: model.safetensors-00001-of-00001.safetensors). The sharded layout must convert
// to exactly the same artifact as the single-file layout.
func TestConvert_ShardedCheckpointMatchesSingleFile(t *testing.T) {
	single := t.TempDir()
	createSyntheticSafetensors(t, single)
	cfg := `{"vocab_size": 4}`
	tok := `{"model": {"vocab": {"a": 0, "b": 1, "c": 2, "d": 3}, "merges": ["a b"]}}`
	for _, dir := range []string{single} {
		os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0644)
		os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tok), 0644)
	}
	opts := Options{ID: "m", Version: 1, Quant: QuantInt8Block32}
	want, _, err := Convert(single, opts)
	if err != nil {
		t.Fatal(err)
	}

	sharded := t.TempDir()
	data, _ := os.ReadFile(filepath.Join(single, SingleFileName))
	shard := "model.safetensors-00001-of-00001.safetensors"
	os.WriteFile(filepath.Join(sharded, shard), data, 0644)
	os.WriteFile(filepath.Join(sharded, "config.json"), []byte(cfg), 0644)
	os.WriteFile(filepath.Join(sharded, "tokenizer.json"), []byte(tok), 0644)
	st, err := OpenSafetensors(filepath.Join(sharded, shard))
	if err != nil {
		t.Fatal(err)
	}
	wm := map[string]string{}
	for name := range st.Tensors {
		wm[name] = shard
	}
	st.Close()
	idx, _ := json.Marshal(map[string]any{"weight_map": wm})
	os.WriteFile(filepath.Join(sharded, IndexFileName), idx, 0644)

	got, _, err := Convert(sharded, opts)
	if err != nil {
		t.Fatalf("sharded Convert: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("sharded checkpoint must produce the same artifact as the single file")
	}
}

func TestConvert_NoCheckpointNamesBothLayouts(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"vocab_size": 1}`), 0644)
	os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"model": {"vocab": {"a": 0}, "merges": []}}`), 0644)
	_, _, err := Convert(dir, Options{ID: "m", Version: 1})
	if err == nil || !strings.Contains(err.Error(), SingleFileName) || !strings.Contains(err.Error(), IndexFileName) {
		t.Fatalf("error = %v, want it to name both layouts", err)
	}
}
