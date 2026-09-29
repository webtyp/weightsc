package weightsc

import (
	"math"
	"math/rand"
	"testing"
)

func TestQuantizeBlocksInt8_RoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	row := make([]float32, 40)
	for i := range row {
		row[i] = (rng.Float32()*2 - 1) * 5.0
	}

	q, scales := QuantizeBlocksInt8(row)

	if len(q) != 40 {
		t.Fatalf("len(q) = %d, want 40", len(q))
	}
	if len(scales) != 2 {
		t.Fatalf("len(scales) = %d, want 2", len(scales))
	}

	for i, v := range row {
		blockIdx := i / 32
		scale := scales[blockIdx]
		dequant := float32(q[i]) * scale
		diff := math.Abs(float64(dequant - v))
		tol := float64(scale/2.0) + 1e-5
		if diff > tol {
			t.Errorf("row[%d] = %v, q = %d, scale = %v, dequant = %v, diff = %v > tol %v",
				i, v, q[i], scale, dequant, diff, tol)
		}
	}
}

func TestQuantizeBlocksInt8_ZeroBlock(t *testing.T) {
	row := make([]float32, 32)
	q, scales := QuantizeBlocksInt8(row)

	if len(scales) != 1 || scales[0] != 1.0 {
		t.Errorf("scales = %v, want [1.0]", scales)
	}
	for i, v := range q {
		if v != 0 {
			t.Errorf("q[%d] = %d, want 0", i, v)
		}
	}
}
