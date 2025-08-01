# Smoothness Really Matters: A Simple Yet Effective Approach for Unsupervised Graph Domain Adaptation
This is the source code of AAAI-2025 paper "[Smoothness Really Matters: A Simple Yet Effective Approach for Unsupervised Graph Domain Adaptation](https://arxiv.org/abs/2412.11654)" (TDSS).
We made modifications based on this code: https://github.com/Meihan-Liu/24AAAI-A2GNN

# Requirements
This code requires the following:
* torch==1.11.0
* torch-scatter==2.0.9
* torch-sparse==0.6.13
* torch-cluster==1.6.0
* torch-geometric==2.1.0
* numpy==1.19.2
* scikit-learn==0.24.2

# Dataset
Datasets used in the paper are all publicly available datasets. You can find [Twitch](https://github.com/benedekrozemberczki/datasets#twitch-social-networks) and [Citation](https://github.com/yuntaodu/ASN/tree/main/data) via the links.

# Cite
If you compare with, build on, or use aspects of framework, please consider citing the following paper:
```
@article{chen2025smoothness,
  title={Smoothness Really Matters: A Simple yet Effective Approach for Unsupervised Graph Domain Adaptation},
  author={Chen, Wei and Ye, Guo and Wang, Yakun and Zhang, Zhao and Zhang, Libang and Wang, Daxin and Zhang, Zhiqiang and Zhuang, Fuzhen},
  journal={AAAI},
  year={2025}
}
``` 
