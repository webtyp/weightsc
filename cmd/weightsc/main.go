package main

import (
	"flag"
	"fmt"
	"os"

	"webtyp.com/weightsc"
)

func printUsage() {
	fmt.Println("Usage: weightsc -in <dir> -out <file.wtypw> -merges-out <file.merges> -id <artifact-id> -version <uint32>")
	flag.PrintDefaults()
}

func main() {
	if len(os.Args) <= 1 {
		printUsage()
		os.Exit(0)
	}

	inDir := flag.String("in", "", "input directory containing model.safetensors, config.json, and tokenizer.json")
	outFile := flag.String("out", "", "output .wtypw artifact file path")
	mergesOutFile := flag.String("merges-out", "", "output .merges companion file path")
	artifactID := flag.String("id", "", "artifact ID")
	version := flag.Uint("version", 0, "artifact version number")

	flag.Usage = func() {
		printUsage()
	}

	flag.Parse()

	if *inDir == "" || *outFile == "" || *mergesOutFile == "" || *artifactID == "" || *version == 0 {
		fmt.Fprintln(os.Stderr, "Error: missing or invalid required flags (-in, -out, -merges-out, -id, -version)")
		printUsage()
		os.Exit(1)
	}

	artifactBytes, mergesBytes, err := weightsc.Convert(*inDir, *artifactID, uint32(*version))
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
