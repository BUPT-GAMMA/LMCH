# LMCH
The source code of Harnessing Language Model for Cross-Heterogeneity Graph Knowledge Transfer.

The source code is based on [LMBot](https://github.com/czjdsg/LMBot)

## Requirements and Installation
Run following command to create environment for reproduction (for cuda 10.2):
```
conda env create -f LMCH.yaml
conda activate LMCH
pip install torch==1.12.0+cu102 torchvision==0.13.0+cu102 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu102
```
For ```pyg_lib```, ```torch_cluster```, ```torch_scatter```, ```torch_sparse``` and ```torch_spline_conv```, please download [here](https://data.pyg.org/whl/torch-1.12.0%2Bcu102.html) and install locally.
```
pip install pyg_lib-0.1.0+pt112cu102-cp39-cp39-linux_x86_64.whl torch_cluster-1.6.0+pt112cu102-cp39-cp39-linux_x86_64.whl torch_scatter-2.1.0+pt112cu102-cp39-cp39-linux_x86_64.whl torch_sparse-0.6.16+pt112cu102-cp39-cp39-linux_x86_64.whl torch_spline_conv-1.2.1+pt112cu102-cp39-cp39-linux_x86_64.whl
```

## Metapath-Based Corpus Construction
Run the following commands to construct metapath-based corpus for ```DBLP```:
```
python preprocess_DBLP.py
python metapath_based_corpus_construction.py
```

## Data Preperation
We have placed the processed data for fine-tuning and GNN-supervised LM training in the ```data``` folder. If you wish to conduct experiments on your own dataset, you can refer to the implementation provided in ```process_target_dataset.py```.


## LM Cross-Heterogeneity Fine-Tuning
Before training, you need to download the pretrained [LM](https://huggingface.co/distilbert/distilroberta-base/tree/main) to the local folder ```distilroberta-base```.

Then, run the following commands to fine-tune LM:
```
python fine_tuning_LM.py --LM_output_size 27
```


## GNN-Supervised LM Fine-Tuning and LM-GNN Contrastive Alignment
Run the following commands to realize GNN-supervised LM training and LM-GNN contrastive alignment:
```
python GNN-supervised_LM_training.py --LM_output_size 3
```