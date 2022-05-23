# Sampple Information Works

## 0. Requirements
The code was fully tested on Ubuntu 18.04 with CUDA 11.3, GCC 7.3, Python 3.7, Pytorch 1.10.2 and OpenCV 3.4.2.

## 1. Introduction
Data has now become a shortcoming of deep learning. Researchers in their own fields share the thinking that "neural networks might not always perform better when they eat more data," which still lacks experimental validation and a convincing guiding theory. Here to fill this lack, we design experiments from Identically Independent Distribution(IID) and Out of Distribution(OOD), which give powerful answers. 

For the purpose of guidance, based on the discussion of results, two theories are proposed: under IID condition, the amount of information determines the effectivity of each sample, the contribution of samples and difference between classes determine the amount of sample information and the amount of class information.

Under OOD condition, the cross-domain degree of samples determine the contributions, and the bias-fitting caused by irrelevant elements is a significant factor of cross-domain. The above theories provide guidance from the perspective of data, which can promote a wide range of practical applications of artificial intelligence.

## 2. Code Composition
This section will describe the makeup of our working code. First, our work is divided into four parts: IID/OOD sample addition and reduction experiments, sample selection strategy design based on two-tier amount of information theory under IID conditions, and element-based OOD sample analysis. Among them, the codes of the first part and the third part are combined.

### [2.1 IID Experiments and Strategies][1]
[1]: ./IID_Experiments_and_Strategies/readme.md

### [2.2 OOD Experiments][2]
[2]: ./OOD_Experiments/readme.md

### [2.3 OOD Analysis: bias-fitting][3]
[3]: ./OOD_Analysis/readme.md

## 3. Papers and Citations


