# open-problems-multimodal
Modified 1st place solution of Kaggle Open Problems - Multimodal Single-Cell Integration, I add ESM-2 embedder, flowformer encoder, and VAE latent.

## Preparation
Install the solution code.
```shell
pip3 install -e .
```

In addtion, download the following data
1. Open Problems - Multimodal Single-Cell Integration data set from Kaggle
2. tab separated hgnc_complete_set file from https://www.genenames.org/download/archive/
3. Reactome Pathways Gene Set from https://reactome.org/download-data

## Compress data and make addtitional data
compress kaggle dataset and make addtional data to use in training
```shell
python3 script/make_compressed_dataset.py
python3 script/make_additional_files.py
python3 script/make_cite_input_mask.py
```

## Training

### Cite
```shell
python3 scripts/train_mode.py
```
