# BlindSelfPortrait

This repository now contains the current Python pipeline for converting black
line-art images into a single continuous vector path for plotting.

## Contents

- `continuous_vector_line.py`: dependency-light vectorization and routing pipeline.
- `ColoringBook/`: source reference images and generated continuous-vector outputs.

The old openFrameworks apps, notebook experiments, CLD binaries, and Potrace
bundles have been removed.

## Usage

```console
python3 continuous_vector_line.py ColoringBook/chatgpt-outline.png
```

By default this writes outputs into `ColoringBook/continuous_vectors/`, including
SVGs, reconstruction images, the fixed 10px/30% opacity render, and a flattened
plotter path JSON.
