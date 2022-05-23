# OOD Experiments and Strategies

## 1. Introduction

Do convolutional neural networks always perform better when eating more data?
OOD Experiments and Strategies related code (Pytorch implementation).

## 2. Code Composition

#### `src/config.py`

The main hyper-parameters.

#### `src/main_exe.py`

To train the feature extractor and models.

#### `src/test_main.py`

To get the embeddings and prototypes, split datasets.

#### `src/utils`

This directory stores multiple function files used in the main program.

#### `Model`

The model files for each training cycle are stored in this directory .

#### `Logs`

The log files of each training cycle are stored in this directory .

## 3.Usage

### Setups

The environment is as bellow:  

- Python >=3.7
- Pytorch >=1.6

### Train the Feature Extractor

Complete the `./src/config.py`.

```python
batch_size =  50 
mission = 'classification'   #extraction, classification
EPOCH = 100
num_workers = 8
lr_list = [0.01]  

dataset_names = ['Your Dataset]

classifier_names = ['ResNet18']    #VGG16、ResNet18、WRN-22-8
```

Then run `./src/main_exe.py` and save the model in `./Model`. Remember to change your own dataset folder path in `./src/utils/dataset_select.py`.

### Get the feature prototypes

Change the `dataset_names` in `./src/config.py`, and set  `batch_size = 1`, `mission = 'extraction'`.  Run `./src/test_main.py`  *"Save feature prototype" part*.

### Split dataset

 Change the `dataset_names` in `./src/config.py`, and set  `batch_size = 1`, `mission = 'extraction'`.  Run `./src/test_main.py` *"Split dataset" part*.