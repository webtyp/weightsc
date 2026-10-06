package main

import (
	"fmt"
	"webtyp.com/weights"
)

func main() {
    var t weights.Tensor
    row := t.DequantRow(0)
    fmt.Printf("%T\n", row)
}
