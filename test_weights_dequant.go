package main

import (
	"fmt"
	"webtyp.com/weights"
)

func main() {
    t := &weights.Tensor{
        DType: weights.Int4Block32,
    }
    fmt.Printf("%T\n", t.DequantRow)
}
