# BlindSelfPortrait

This repository now contains the current Python pipeline for converting black
line-art images into a single continuous vector path for plotting.

## Contents

- `continuous_vector_line.py`: dependency-light vectorization and routing pipeline.
- `input-coloring/`: source line-art images.
- `output-vectors/`: generated continuous-vector outputs.

The old openFrameworks apps, notebook experiments, CLD binaries, and Potrace
bundles have been removed.

## Usage

```console
python3 continuous_vector_line.py
```

By default this reads images from `input-coloring/` and writes only two outputs
per image into `output-vectors/`: the vectors JSON and the fixed 10px/30%
opacity render.
