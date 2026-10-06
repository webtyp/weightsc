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
	VocabSize  int `json:"vocab_size"`
	TextConfig struct {
		VocabSize int `json:"vocab_size"`
	} `json:"text_config"`
}

type addedToken struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
}

type tokenizerJSON struct {
	Model struct {
		Vocab  map[string]int    `json:"vocab"`
		Merges []json.RawMessage `json:"merges"`
	} `json:"model"`
	AddedTokens []addedToken `json:"added_tokens"`
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

func parseVocab(vocabMap map[string]int, addedTokens []addedToken, vocabSize int) ([]string, error) {
	maxID := -1
	for _, id := range vocabMap {
		if id > maxID {
			maxID = id
		}
	}
	for _, item := range addedTokens {
		if item.ID > maxID {
			maxID = item.ID
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

	for _, item := range addedTokens {
		id := item.ID
		tok := item.Content
		if id >= 0 && id < size {
			if filled[id] && vocab[id] != tok {
				return nil, fmt.Errorf("weightsc: token id %d is both %q and %q", id, vocab[id], tok)
			}
			vocab[id] = tok
			filled[id] = true
		}
	}

	for id := 0; id < size; id++ {
		if !filled[id] {
			vocab[id] = fmt.Sprintf("<|pad_%d|>", id)
		}
	}

	return vocab, nil
}

// Convert reads a model directory containing config.json, tokenizer.json, and model.safetensors
// (or shards listed in model.safetensors.index.json),
// and produces the WTYPW1 artifact bytes and .merges content.
func Convert(inDir string, opts Options) ([]byte, []byte, error) {
	if err := opts.Validate(); err != nil {
		return nil, nil, err
	}

	cfgPath := filepath.Join(inDir, "config.json")
	cfgBytes, err := os.ReadFile(cfgPath)
	if err != nil {
		return nil, nil, fmt.Errorf("reading config.json: %w", err)
	}
	var cfg configJSON
	if err := json.Unmarshal(cfgBytes, &cfg); err != nil {
		return nil, nil, fmt.Errorf("parsing config.json: %w", err)
	}

	vocabSize := cfg.VocabSize
	if vocabSize == 0 {
		vocabSize = cfg.TextConfig.VocabSize
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

	vocab, err := parseVocab(tokData.Model.Vocab, tokData.AddedTokens, vocabSize)
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

	st, err := openCheckpoint(inDir)
	if err != nil {
		return nil, nil, err
	}
	defer st.Close()

	names := make([]string, 0)
	for _, name := range st.names() {
		if opts.Prefix != "" && !strings.HasPrefix(name, opts.Prefix) {
			continue
		}
		names = append(names, name)
	}
	sort.Strings(names)

	quant := opts.Quant
	if quant == "" {
		quant = QuantInt8Row
	}

	inputs := make([]weights.TensorInput, 0, len(names))
	for _, name := range names {
		f32s, shape, err := st.readFloat32(name)
		if err != nil {
			return nil, nil, fmt.Errorf("reading tensor %s: %w", name, err)
		}

		expectedLen := 1
		for _, d := range shape {
			expectedLen *= d
		}
		if len(f32s) != expectedLen {
			return nil, nil, fmt.Errorf("tensor %s size mismatch: shape %v vs data len %d", name, shape, len(f32s))
		}

		if len(shape) == 2 {
			rows := shape[0]
			cols := shape[1]

			switch quant {
			case QuantInt8Row:
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

			case QuantInt8Block32:
				inputs = append(inputs, int8Block32Input(name, shape, f32s))

			case QuantInt4Block32:
				if cols%weights.BlockSize != 0 {
					inputs = append(inputs, int8Block32Input(name, shape, f32s))
					break
				}
				data := make([]byte, 0, rows*cols/2)
				scales := make([]float32, 0, rows*cols/weights.BlockSize)
				for r := 0; r < rows; r++ {
					q, s, err := weights.QuantizeInt4Block32(f32s[r*cols : (r+1)*cols])
					if err != nil {
						return nil, nil, fmt.Errorf("tensor %s: %w", name, err)
					}
					data = append(data, q...)
					scales = append(scales, s...)
				}
				inputs = append(inputs, weights.TensorInput{Name: name, DType: weights.Int4Block32, Shape: shape, Data: data, Scales: scales})

			case QuantFloat32:
				data := make([]byte, len(f32s)*4)
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
			}
		} else {
			// 1-D and >= 3-D tensors: always float32, shape kept as in the file
			data := make([]byte, len(f32s)*4)
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
		}
	}

	artifactBytes, err := weights.WriteArtifact(opts.ID, opts.Version, tokConfig, inputs)
	if err != nil {
		return nil, nil, fmt.Errorf("writing artifact: %w", err)
	}

	return artifactBytes, mergesBuf.Bytes(), nil
}

// int8Block32Input quantizes a 2-D tensor to int8 with one scale per weights.BlockSize values of
// each row.
func int8Block32Input(name string, shape []int, f32s []float32) weights.TensorInput {
	rows, cols := shape[0], shape[1]
	numBlocksPerRow := (cols + weights.BlockSize - 1) / weights.BlockSize
	scales := make([]float32, 0, rows*numBlocksPerRow)
	qData := make([]byte, 0, rows*cols)
	for r := 0; r < rows; r++ {
		qRow, rowScales := QuantizeBlocksInt8(f32s[r*cols : (r+1)*cols])
		scales = append(scales, rowScales...)
		for _, qVal := range qRow {
			qData = append(qData, byte(qVal))
		}
	}
	return weights.TensorInput{Name: name, DType: weights.Int8Block32, Shape: shape, Data: qData, Scales: scales}
}
