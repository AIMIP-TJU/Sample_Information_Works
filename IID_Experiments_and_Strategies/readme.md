# IID Experiments and Strategies

## 1. Introduction
Do convolutional neural networks always perform better when eating more data? 

IID Experiments and Strategies related code(Pytorch implementation).

## 2. Code Composition
#### `Requirements.txt`

This directory is used to store package files required for installation.

#### `ADD_main.py`

"Addition Experiment"   main program.

#### `REDUCE_main.py`

"Reduction Experiment"  main program.

#### `dataset_select.py`

Create your datasets and write the dataset related information.

#### `src/classifier`

This directory stores different backbone source files.

#### `src/my_select`

This directory stores multiple source files of information evaluation indicator functions.

#### `src/utils`

This directory stores multiple function files used in the main program.

#### `Selcetion`

The sample index files selected for each training cycle are stored in this directory .

#### `Result`

The results of the experiment records are stored in this directory .

#### `Model`

The model files for each training cycle are stored in this directory .

#### `Logs`

The log files of each training cycle are stored in this directory .

## 3.Usage

### Setups

  All code was developed and tested on a single machine equiped with a  NVIDIA RTX 3080Ti GPU. The environment is as bellow:  

- Ubuntu 18.04
- CUDA 11.3
- Python 3.7
- Pytorch 1.10.2

### Install requirements

```
pip install -r requirements.txt
```

### Create your datasets

Edit `dataset_select.py` and write the dataset related information in the following format.

```python
if dataset_name == 'Your dataests name':
    file_Path = 'The path of your datasets images'
    train_name = 'The path of your train set index(CSV)'
    test_name = 'The path of your test set index(CSV)'
    num_classes = 10 #The number of classes
    num_input = 3 #The dim of images
```

### Running IID on benchmark datasets (CIFAR-10 and Mini-ImageNet)

Here is an "Addition Experiment" example: 

```
python ADD_main.py --dataset ['cifar10','Mini-ImageNet']  --device 'cuda:1' --method_names metric_select
```

Here is an "Reduction Experiment" example: 

```
python REDUCE_main.py --dataset ['cifar10','Mini-ImageNet']  --device 'cuda:1' --method_names metric_select
```

