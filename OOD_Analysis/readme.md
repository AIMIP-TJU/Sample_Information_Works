# OOD Analysis: bias-fitting

##1. Introduction
Data has now become a shortcoming of deep learning. Researchers in their own fields share the thinking that "neural networks might not always perform better when they eat more data," which still lacks experimental validation and a convincing guiding theory. Here to fill this lack, we design experiments from Identically Independent Distribution(IID) and Out of Distribution(OOD), which give powerful answers. 

For the purpose of guidance, based on the discussion of results, two theories are proposed: under IID condition, the amount of information determines the effectivity of each sample, the contribution of samples and difference between classes determine the amount of sample information and the amount of class information.

Under OOD condition, the cross-domain degree of samples determine the contributions, and the bias-fitting caused by irrelevant elements is a significant factor of cross-domain. The above theories provide guidance from the perspective of data, which can promote a wide range of practical applications of artificial intelligence.

##2. Code Composition
This section will describe the makeup of our working code. First, our work is divided into four parts: IID/OOD sample addition and reduction experiments, sample selection strategy design based on two-tier amount of information theory under IID conditions, and element-based OOD sample analysis. Among them, the codes of the first part and the third part are combined.

##3. Usage

