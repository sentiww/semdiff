# semdiff

`semdiff` evaluates image-classification models on ImageNet datasets and then computes semantic metrics over the predictions.

The project currently supports:

- datasets: `imagenet-1k`, `imagenet-o`
- models: `resnet`, `densenet`, `vgg`, `vit-b-16`, `clip-vit-b-16` (experimental)
- analysis: WordNet based semantic metrics

## What The Project Does

The workflow has two stages:

1. Run model evaluation on a dataset.
2. Run semantic analysis on the saved predictions.

Evaluation writes raw outputs to:

```text
output/raw/<model>/<dataset>/
```

Analysis reads those raw predictions and writes processed outputs to:

```text
output/processed/<model>/<dataset>/
```

## Setup

### Clone the repository

```bash
git clone https://github.com/sentiww/semdiff.git
cd semdiff
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install the shell alias for this session

Source `activate.sh` to define `semdiff` in your current shell:

```bash
source activate.sh
```

After that, you can run commands like:

```bash
semdiff wordnet init
semdiff evaluate resnet imagenet-o
```

If you prefer, you can still run commands directly with:

```bash
python src/main.py ...
```

or:

```bash
./src/semdiff.sh ...
```

## Datasets

Datasets live under:

```text
datasets/
```

### ImageNet-1k

Place these files in:

```text
datasets/imagenet-1k/
```

Required files:

- `ILSVRC2012_img_val.tar`
- `ILSVRC2012_validation_ground_truth.txt`
- `meta.mat`

Then initialize synset folders:

```bash
semdiff datasets init imagenet-1k
```

This reorganizes the validation images into folders like:

```text
datasets/imagenet-1k/n01440764/
datasets/imagenet-1k/n01443537/
...
```

### ImageNet-O

ImageNet-O is downloaded automatically, but WordNet must be initialized first:

```bash
semdiff wordnet init
semdiff datasets init imagenet-o
```

## Commands

### WordNet

Initialize the local WordNet corpus used by the semantic analysis tools:

```bash
semdiff wordnet init
```

### Dataset commands

Initialize a dataset:

```bash
semdiff datasets init imagenet-1k
semdiff datasets init imagenet-o
```

Clear generated synset folders for a dataset:

```bash
semdiff datasets clear imagenet-1k
semdiff datasets clear imagenet-o
```

### Evaluation commands

Run evaluation for a model and dataset:

```bash
semdiff evaluate resnet imagenet-o
```

### Analysis commands

Build semantic metrics from raw prediction outputs:

```bash
semdiff analysis semantic resnet imagenet-o
```

## Output Layout

### Raw evaluation output

Evaluation writes to:

```text
output/raw/<model>/<dataset>/
```

Files:

- `predictions.jsonl`
- `summary.json`

`predictions.jsonl` contains one record per sample. Each record includes:

- `id`
- `image`
- `target_synset`
- `predicted_synset`
- `predicted_label`
- `confidence`

### Processed analysis output

Analysis writes to:

```text
output/processed/<model>/<dataset>/
```

Files:

- `semantics.jsonl`
- `semantic-summary.json`

`semantics.jsonl` contains one record per sample. Each record includes:

- `id`
- `path_distance`
- `path_similarity`
- `wup_similarity`

`semantic-summary.json` contains aggregate statistics and example extremes for each metric.

## Synset Utilities

Get the synset id for a label:

```bash
semdiff synset id goldfish
```

Get readable labels for a synset id:

```bash
semdiff synset readable n01443537
```

## Notes

- `activate.sh` only affects the current shell session because it defines an alias.
- `imagenet-o` setup downloads the archive and reorganizes images into WordNet synset folders.
