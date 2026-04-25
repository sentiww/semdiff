# semdiff

`semdiff` evaluates image-classification models on ImageNet datasets and then computes semantic metrics over the predictions.

The project currently supports:

- datasets: `imagenet-1k`, `imagenet-o`
- models: `resnet`, `densenet`, `vgg`, `vit-b-16`, `clip-vit-b-16` (experimental)
- analysis: WordNet based semantic metrics (path_distance, path_similarity, wup_similarity, lch_similarity, jcn_similarity, lin_similarity, res_similarity)

## What The Project Does

The workflow has two stages:

1. Run model evaluation on a dataset.
2. Run semantic analysis on the saved predictions.

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
semdiff evaluate run --model resnet --input datasets/imagenet-o --output output/resnet-o
```

If you prefer, you can still run commands directly with:

```bash
python src/__main__.py evaluate run --model resnet --input datasets/imagenet-o --output output/resnet-o
```

or:

```bash
./src/semdiff.sh evaluate run --model resnet --input datasets/imagenet-o --output output/resnet-o
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
semdiff datasets init imagenet-1k --input /path/to/archive.tar --output datasets/imagenet-1k
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
semdiff datasets init imagenet-o --input https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar --output datasets/imagenet-o
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
semdiff datasets init imagenet-1k --input /path/to/val.tar --output datasets/imagenet-1k
semdiff datasets init imagenet-o --input https://example.com/imagenet-o.tar --output datasets/imagenet-o
```

### Evaluation commands

Run evaluation for a model and dataset:

```bash
semdiff evaluate run --model resnet --input datasets/imagenet-o --output output/resnet-o
```

Available models: `resnet`, `densenet`, `vgg`, `vit-b-16`, `clip-vit-b-16`

### Analysis commands

Build semantic metrics from prediction outputs:

```bash
semdiff analysis semantic --metric wup_similarity --input output/resnet-o/predictions.jsonl --output output/resnet-o-analysis
```

Available metrics: `path_distance`, `path_similarity`, `wup_similarity`, `lch_similarity`, `jcn_similarity`, `lin_similarity`, `res_similarity`

## Synset Utilities

Get the synset id for a label:

```bash
semdiff synset id goldfish
```

Get readable labels for a synset id:

```bash
semdiff synset readable n01443537
```

### Visualization commands

Compare metric distributions across models or datasets:

```bash
semdiff visualization distribution --input output/resnet-o-analysis/semantic-wup_similarity.json --output output/distribution.png
```

## Notes

- `activate.sh` only affects the current shell session because it defines an alias.
- `imagenet-o` setup downloads the archive and reorganizes images into WordNet synset folders.