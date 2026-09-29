package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"webtyp.com/weightsc"
)

func printUsage() {
	fmt.Println("Usage: weightsc -in <dir> -out <file.wtypw> -merges-out <file.merges> -id <artifact-id> -version <uint32> [-quant int8-row|int8-block32|float32] [-prefix <tensor name prefix>]")
	flag.PrintDefaults()
}

func main() {
	inDir := flag.String("in", "", "input directory containing model.safetensors, config.json, and tokenizer.json")
	outFile := flag.String("out", "", "output .wtypw artifact file path")
	mergesOutFile := flag.String("merges-out", "", "output .merges companion file path")
	artifactID := flag.String("id", "", "artifact ID")
	version := flag.Uint("version", 0, "artifact version number")
	quant := flag.String("quant", "int8-row", "quantization for 2-D tensors (int8-row, int8-block32, float32)")
	prefix := flag.String("prefix", "", "filter tensors by name prefix")

	flag.Usage = func() {
		printUsage()
	}

	if len(os.Args) <= 1 {
		printUsage()
		os.Exit(0)
	}

	flag.Parse()

	if *version > math.MaxUint32 {
		fmt.Fprintf(os.Stderr, "Error: -version %d exceeds uint32 range (max %d)\n", *version, uint32(math.MaxUint32))
		os.Exit(1)
	}

	opts := weightsc.Options{
		ID:      *artifactID,
		Version: uint32(*version),
		Quant:   weightsc.Quant(*quant),
		Prefix:  *prefix,
	}

	if err := opts.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		printUsage()
		os.Exit(1)
	}

	if *inDir == "" || *outFile == "" || *mergesOutFile == "" {
		fmt.Fprintln(os.Stderr, "Error: missing required file path flags (-in, -out, -merges-out)")
		printUsage()
		os.Exit(1)
	}

	artifactBytes, mergesBytes, err := weightsc.Convert(*inDir, opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	if err := os.WriteFile(*outFile, artifactBytes, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing output artifact: %v\n", err)
		os.Exit(1)
	}

	if err := os.WriteFile(*mergesOutFile, mergesBytes, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing merges file: %v\n", err)
		os.Exit(1)
	}
}
