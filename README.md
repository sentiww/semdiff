# semdiff

## imagenet-1k

Download imagenet-1k 2012 files: Development kit, Validation images
Move files ILSVRC2012_img_val.tar, ILSVRC2012_validation_ground_truth.txt, meta.mat to datasets/imagenet-1k

Initialize the dataset into synset folders:

```bash
./run.sh datasets init imagenet-1k
```

## imagenet-o

Initialize the dataset (no need to download manually):

```bash
./run.sh wordnet init
./run.sh datasets init imagenet-o
```
