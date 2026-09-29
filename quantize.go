package weightsc

import (
	"math"

	"webtyp.com/weights"
)

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

// QuantizeBlocksInt8 converts one row of float32 values to int8 with one scale per block of weights.BlockSize values.
// For each block of weights.BlockSize values (the last block may be shorter):
// scale = max|x| / 127 (a zero block gets scale 1); q = round(x / scale) clamped to [-127, 127].
func QuantizeBlocksInt8(row []float32) ([]int8, []float32) {
	if len(row) == 0 {
		return nil, nil
	}

	q := make([]int8, 0, len(row))
	numBlocks := (len(row) + weights.BlockSize - 1) / weights.BlockSize
	scales := make([]float32, 0, numBlocks)

	for i := 0; i < len(row); i += weights.BlockSize {
		end := i + weights.BlockSize
		if end > len(row) {
			end = len(row)
		}
		block := row[i:end]

		var maxAbs float32
		for _, v := range block {
			absVal := v
			if absVal < 0 {
				absVal = -absVal
			}
			if absVal > maxAbs {
				maxAbs = absVal
			}
		}

		if maxAbs == 0 {
			scales = append(scales, 1.0)
			q = append(q, make([]int8, len(block))...)
			continue
		}

		scale := maxAbs / 127.0
		scales = append(scales, scale)

		for _, v := range block {
			scaled := math.Round(float64(v / scale))
			if scaled > 127 {
				scaled = 127
			} else if scaled < -127 {
				scaled = -127
			}
			q = append(q, int8(scaled))
		}
	}

	return q, scales
}
