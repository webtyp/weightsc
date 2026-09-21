package weightsc

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"webtyp.com/weights"
)

type configJSON struct {
	VocabSize int `json:"vocab_size"`
}

type tokenizerJSON struct {
	Model struct {
		Vocab  map[string]int    `json:"vocab"`
		Merges []json.RawMessage `json:"merges"`
	} `json:"model"`
}

// QuantizeRowInt8 converts one row of float32 values to int8 with a per-row scale.
// scale = max(abs(row)) / 127; q[i] = round(row[i] / scale), clamped to [-127, 127].
// A row of all zeros gets scale = 1 (avoid divide by zero) and stays all zeros.
func QuantizeRowInt8(row []float32) ([]int8, float32) {
	if len(row) == 0 {
		return nil, 1.0
	}
	var maxAbs float32
	for _, v := range row {
		absVal := v
		if absVal < 0 {
			absVal = -absVal
		}
		if absVal > maxAbs {
			maxAbs = absVal
		}
	}

	if maxAbs == 0 {
		return make([]int8, len(row)), 1.0
	}

	scale := maxAbs / 127.0
	q := make([]int8, len(row))
	for i, v := range row {
		scaled := math.Round(float64(v / scale))
		if scaled > 127 {
			scaled = 127
		} else if scaled < -127 {
			scaled = -127
		}
		q[i] = int8(scaled)
	}

	return q, scale
}

func parseMerges(rawMerges []json.RawMessage) ([]string, error) {
	merges := make([]string, 0, len(rawMerges))
	for _, raw := range rawMerges {
		var pair []string
		if err := json.Unmarshal(raw, &pair); err == nil && len(pair) == 2 {
			merges = append(merges, pair[0]+" "+pair[1])
			continue
		}
		var str string
		if err := json.Unmarshal(raw, &str); err == nil {
			str = strings.TrimSuffix(str, "\n")
			merges = append(merges, str)
			continue
		}
		return nil, fmt.Errorf("invalid merge element: %s", string(raw))
	}
	return merges, nil
}

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

// Convert reads a model directory containing config.json, tokenizer.json, and model.safetensors,
// and produces the WTYPW1 artifact bytes and .merges content.
func Convert(inDir string, artifactID string, version uint32) ([]byte, []byte, error) {
	cfgPath := filepath.Join(inDir, "config.json")
	cfgBytes, err := os.ReadFile(cfgPath)
	if err != nil {
		return nil, nil, fmt.Errorf("reading config.json: %w", err)
	}
	var cfg configJSON
	if err := json.Unmarshal(cfgBytes, &cfg); err != nil {
		return nil, nil, fmt.Errorf("parsing config.json: %w", err)
	}

	tokPath := filepath.Join(inDir, "tokenizer.json")
	tokBytes, err := os.ReadFile(tokPath)
	if err != nil {
		return nil, nil, fmt.Errorf("reading tokenizer.json: %w", err)
	}
	var tokData tokenizerJSON
	if err := json.Unmarshal(tokBytes, &tokData); err != nil {
		return nil, nil, fmt.Errorf("parsing tokenizer.json: %w", err)
	}

	vocab, err := parseVocab(tokData.Model.Vocab, cfg.VocabSize)
	if err != nil {
		return nil, nil, fmt.Errorf("parsing vocab from tokenizer.json: %w", err)
	}

	merges, err := parseMerges(tokData.Model.Merges)
	if err != nil {
		return nil, nil, fmt.Errorf("parsing merges from tokenizer.json: %w", err)
	}

	var mergesBuf bytes.Buffer
	for _, m := range merges {
		mergesBuf.WriteString(m)
		mergesBuf.WriteString("\n")
	}

	tokConfig := weights.TokenizerConfig{
		Lowercase:    false,
		StripAccents: false,
		Vocab:        vocab,
	}

	stPath := filepath.Join(inDir, "model.safetensors")
	st, err := OpenSafetensors(stPath)
	if err != nil {
		return nil, nil, fmt.Errorf("opening model.safetensors: %w", err)
	}
	defer st.Close()

	names := make([]string, 0, len(st.Tensors))
	for name := range st.Tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	inputs := make([]weights.TensorInput, 0, len(names))
	for _, name := range names {
		f32s, shape, err := st.ReadTensorFloat32(name)
		if err != nil {
			return nil, nil, fmt.Errorf("reading tensor %s: %w", name, err)
		}

		if len(shape) == 2 {
			rows := shape[0]
			cols := shape[1]
			if len(f32s) != rows*cols {
				return nil, nil, fmt.Errorf("tensor %s size mismatch: shape %v vs data len %d", name, shape, len(f32s))
			}

			qData := make([]byte, rows*cols)
			scales := make([]float32, rows)

			for r := 0; r < rows; r++ {
				rowSlice := f32s[r*cols : (r+1)*cols]
				qRow, scale := QuantizeRowInt8(rowSlice)
				scales[r] = scale
				for c, qVal := range qRow {
					qData[r*cols+c] = byte(qVal)
				}
			}

			inputs = append(inputs, weights.TensorInput{
				Name:   name,
				DType:  weights.Int8,
				Shape:  shape,
				Data:   qData,
				Scales: scales,
			})
		} else if len(shape) == 1 {
			dim := shape[0]
			if len(f32s) != dim {
				return nil, nil, fmt.Errorf("tensor %s size mismatch: shape %v vs data len %d", name, shape, len(f32s))
			}

			data := make([]byte, dim*4)
			for i, v := range f32s {
				binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
			}

			inputs = append(inputs, weights.TensorInput{
				Name:   name,
				DType:  weights.Float32,
				Shape:  shape,
				Data:   data,
				Scales: nil,
			})
		} else {
			return nil, nil, fmt.Errorf("unsupported tensor shape dimension %d for tensor %s", len(shape), name)
		}
	}

	artifactBytes, err := weights.WriteArtifact(artifactID, version, tokConfig, inputs)
	if err != nil {
		return nil, nil, fmt.Errorf("writing artifact: %w", err)
	}

	return artifactBytes, mergesBuf.Bytes(), nil
}
