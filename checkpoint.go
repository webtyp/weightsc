package weightsc

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// SingleFileName and IndexFileName are the two layouts a Hugging Face checkpoint uses: one
// safetensors file, or several shards listed by an index ({"weight_map": {tensor: shard file}}).
const (
	SingleFileName = "model.safetensors"
	IndexFileName  = "model.safetensors.index.json"
)

// checkpoint is every tensor of a model, whether it lives in one safetensors file or in shards.
type checkpoint struct {
	files []*Safetensors
	owner []struct {
		name string
		file int
	}
}

// openCheckpoint opens inDir/model.safetensors, or, when it is absent, every shard listed in
// inDir/model.safetensors.index.json.
func openCheckpoint(inDir string) (*checkpoint, error) {
	single := filepath.Join(inDir, SingleFileName)
	if _, err := os.Stat(single); err == nil {
		st, err := OpenSafetensors(single)
		if err != nil {
			return nil, fmt.Errorf("opening %s: %w", SingleFileName, err)
		}
		return newCheckpoint([]*Safetensors{st}), nil
	}

	raw, err := os.ReadFile(filepath.Join(inDir, IndexFileName))
	if err != nil {
		return nil, fmt.Errorf("no %s and no %s in %s", SingleFileName, IndexFileName, inDir)
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(raw, &idx); err != nil {
		return nil, fmt.Errorf("parsing %s: %w", IndexFileName, err)
	}
	shardNames := make([]string, 0)
	seen := map[string]bool{}
	for _, shard := range idx.WeightMap {
		if !seen[shard] {
			seen[shard] = true
			shardNames = append(shardNames, shard)
		}
	}
	sort.Strings(shardNames)
	var files []*Safetensors
	for _, shard := range shardNames {
		st, err := OpenSafetensors(filepath.Join(inDir, shard))
		if err != nil {
			for _, f := range files {
				f.Close()
			}
			return nil, fmt.Errorf("opening shard %s: %w", shard, err)
		}
		files = append(files, st)
	}
	return newCheckpoint(files), nil
}

func newCheckpoint(files []*Safetensors) *checkpoint {
	c := &checkpoint{files: files}
	for i, f := range files {
		for name := range f.Tensors {
			c.owner = append(c.owner, struct {
				name string
				file int
			}{name, i})
		}
	}
	sort.Slice(c.owner, func(a, b int) bool { return c.owner[a].name < c.owner[b].name })
	return c
}

// names returns every tensor name, sorted.
func (c *checkpoint) names() []string {
	out := make([]string, len(c.owner))
	for i, o := range c.owner {
		out[i] = o.name
	}
	return out
}

// readFloat32 reads one tensor as float32 from whichever file holds it.
func (c *checkpoint) readFloat32(name string) ([]float32, []int, error) {
	i := sort.Search(len(c.owner), func(i int) bool { return c.owner[i].name >= name })
	if i == len(c.owner) || c.owner[i].name != name {
		return nil, nil, fmt.Errorf("tensor %s not found", name)
	}
	return c.files[c.owner[i].file].ReadTensorFloat32(name)
}

func (c *checkpoint) Close() error {
	for _, f := range c.files {
		f.Close()
	}
	return nil
}
