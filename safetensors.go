package weightsc

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

// TensorInfo holds metadata for a tensor stored in a safetensors file.
type TensorInfo struct {
	Name        string
	DType       string  `json:"dtype"`
	Shape       []int   `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
}

// Safetensors handles reading tensors from a safetensors file.
type Safetensors struct {
	Tensors   map[string]TensorInfo
	f         *os.File
	dataStart int64
}

// OpenSafetensors opens a safetensors file and parses its header.
func OpenSafetensors(path string) (*Safetensors, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	var headerLen uint64
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		f.Close()
		return nil, fmt.Errorf("reading header length: %w", err)
	}

	headerBuf := make([]byte, headerLen)
	if _, err := io.ReadFull(f, headerBuf); err != nil {
		f.Close()
		return nil, fmt.Errorf("reading header JSON: %w", err)
	}

	var rawMap map[string]json.RawMessage
	if err := json.Unmarshal(headerBuf, &rawMap); err != nil {
		f.Close()
		return nil, fmt.Errorf("parsing header JSON: %w", err)
	}

	tensors := make(map[string]TensorInfo)
	for name, raw := range rawMap {
		if name == "__metadata__" {
			continue
		}
		var info TensorInfo
		if err := json.Unmarshal(raw, &info); err != nil {
			f.Close()
			return nil, fmt.Errorf("parsing tensor info for %s: %w", name, err)
		}
		info.Name = name
		tensors[name] = info
	}

	dataStart := int64(8 + headerLen)

	return &Safetensors{
		Tensors:   tensors,
		f:         f,
		dataStart: dataStart,
	}, nil
}

// Close closes the underlying safetensors file.
func (s *Safetensors) Close() error {
	if s.f != nil {
		return s.f.Close()
	}
	return nil
}

// ReadTensorFloat32 reads a tensor by name and converts its values to a slice of float32s.
func (s *Safetensors) ReadTensorFloat32(name string) ([]float32, []int, error) {
	info, ok := s.Tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("tensor %s not found", name)
	}

	if len(info.DataOffsets) != 2 {
		return nil, nil, fmt.Errorf("tensor %s invalid data_offsets", name)
	}

	start := info.DataOffsets[0]
	end := info.DataOffsets[1]
	length := end - start

	buf := make([]byte, length)
	_, err := s.f.ReadAt(buf, s.dataStart+start)
	if err != nil {
		return nil, nil, fmt.Errorf("reading tensor %s data: %w", name, err)
	}

	switch info.DType {
	case "BF16":
		if length%2 != 0 {
			return nil, nil, fmt.Errorf("tensor %s BF16 length not multiple of 2", name)
		}
		count := int(length / 2)
		f32s := make([]float32, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint16(buf[i*2:])
			f32s[i] = bf16ToF32(bits)
		}
		return f32s, info.Shape, nil
	case "F32":
		if length%4 != 0 {
			return nil, nil, fmt.Errorf("tensor %s F32 length not multiple of 4", name)
		}
		count := int(length / 4)
		f32s := make([]float32, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint32(buf[i*4:])
			f32s[i] = math.Float32frombits(bits)
		}
		return f32s, info.Shape, nil
	default:
		return nil, nil, fmt.Errorf("unsupported dtype %s for tensor %s", info.DType, name)
	}
}

// bf16ToF32 converts a uint16 bfloat16 representation to float32.
func bf16ToF32(bits uint16) float32 {
	return math.Float32frombits(uint32(bits) << 16)
}
