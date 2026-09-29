package weightsc

import (
	"fmt"
)

// Quant is how 2-D tensors are stored in the artifact.
type Quant string

const (
	QuantInt8Row     Quant = "int8-row"     // one scale per row (default; the embedding model)
	QuantInt8Block32 Quant = "int8-block32" // one scale per 32 values (language models)
	QuantFloat32     Quant = "float32"      // no quantization (verification, small models)
)

// Options configures one conversion.
type Options struct {
	ID      string // artifact ID, required
	Version uint32 // artifact version, required (> 0)
	Quant   Quant  // "" means QuantInt8Row
	Prefix  string // when set, only tensors whose name starts with Prefix are converted (names are kept whole)
}

// Validate validates that Options fields are properly configured.
func (o Options) Validate() error {
	if o.ID == "" {
		return fmt.Errorf("weightsc: Options.ID is required")
	}
	if o.Version == 0 {
		return fmt.Errorf("weightsc: Options.Version must be greater than zero")
	}

	q := o.Quant
	if q == "" {
		q = QuantInt8Row
	}

	switch q {
	case QuantInt8Row, QuantInt8Block32, QuantFloat32:
		return nil
	default:
		return fmt.Errorf("weightsc: unknown quantization %q (want int8-row, int8-block32 or float32)", o.Quant)
	}
}
