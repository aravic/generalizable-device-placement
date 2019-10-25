# Datasets
We release the following three datasets each consisting of 32 graphs.
They are named as follows:

 - cifar10
 - ptb
 - nmt

### Dataset dependencies
Reading the datasets requires installing the following dependencies
 - python3
 - pickle
 - tensorflow
 - networkx

### Dataset format
Each `input.pkl` file is encoded using pickle version '3'  available as part of standard package with python3

To read use the following python code snippet:
```
import pickle

with open('input.pkl', 'rb') as f:
  j = pickle.load(f)

# j['optim_mg'] => Optimized tensorflow metagraph
# j['op_perf'] & j['step_stats'] => Measurements captured on two K80 GPU configuration available with p2.8xlarge instances on EC2
```

### Code to generate dataset.
We use the open source implementation of ENAS algorithm from [here](https://arxiv.org/abs/1802.03268), which you can find on github:
[https://github.com/melodyguan/enas](https://github.com/melodyguan/enas)


### Download Datasets
Datasets can be downloaded using the following link:
https://www.dropbox.com/sh/7np79c92y006a7c/AABmcOhdKM1GPzmizpXwv9EIa?dl=0
