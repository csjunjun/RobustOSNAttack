# RobustOSNAttack

An official implementation code for paper "Generating Robust Adversarial Examples against Online Social Networks (OSNs)"

## Table of Contents

- [Background](#background)
- [Dependency](#dependency)
- [Demo](#demo)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)


## Background

<p align='center'>  
  <img src='https://github.com/csjunjun/RobustOSNAttack/blob/main/storyqq.jpg' width='500'/>
</p>
<p align='center'>  
  <em>The scenario of transmitting AEs over OSNs and the impact on their attack capabilities.</em>
</p>

Online Social Networks (OSNs) have blossomed into prevailing transmission channels for images in the modern era. Adversarial examples (AEs) deliberately designed to mislead deep neural networks (DNNs) are found to be fragile against the inevitable lossy operations conducted by OSNs. As a result, the AEs would lose their attack capabilities after being transmitted over OSNs. In this work, we aim to design a new framework for generating robust AEs that can survive the OSN transmission; namely, the AEs before and after the OSN transmission both possess strong attack capabilities. To this end, we first propose a differentiable network termed SImulated OSN (SIO) to simulate the various operations conducted by an OSN. Specifically, the SIO network consists of two modules: 1) a differentiable JPEG layer for approximating the ubiquitous JPEG compression and 2) an encoder-decoder subnetwork for mimicking the remaining operations. Based upon the SIO network, we then formulate an optimization framework to generate robust AEs by enforcing model outputs with and without passing through the SIO to be both misled. Extensive experiments conducted over Facebook, WeChat and QQ demonstrate that our attack methods produce more robust AEs than existing approaches, especially under small distortion constraints; the performance gain in terms of Attack Success Rate (ASR) could be more than 60%. Furthermore, we build a public dataset containing more than 10,000 pairs of AEs processed by Facebook, WeChat or QQ, facilitating future research in the robust AEs generation.


## Dependency
- torch 1.10.1
- torchattacks 2.14.0
- python 3.8

## Demo

1. train your SIO model
```bash
python train.py 
```
2. generate robust AEs with the pre-trained SIO model
```bash
python rattacks.py --attype 0
```
**Note: The pretrained weights and dataset can be downloaded from:
[Google Drive](https://drive.google.com/drive/folders/1M-yrL-DDvNd-KV9vxmxxAwAYTVMa6pde?usp=sharing) or 
[Baidu Yun (Code: djzz)](https://pan.baidu.com/s/10GHrwNv57L2d2bD_5X3W0g)**

## Citation

## Acknowledgments
Our code benefits from the [DiffJPEG](https://github.com/mlomnitz/DiffJPEG).



