# Mixup Without Hesitation
This repo contains demo reimplementations of the CIFAR-100 training code in PyTorch based on the following paper:

Hao Yu, Huanyu Wang, Jianxin Wu. *Mixup Without Hesitation.* 

## How to use

```
git clone https://github.com/yuhao318/mwh.git
cd mwh
CUDA_VISIBLE_DEVICES=0 python easy_mwh.py --sess mwh_shufflenetv2_0.5 --alpha 0.5
```

## Results
The following table shows accuracy with $\alpha = 0.5$  and 100 epochs  in CIFAR-100:

| Model   | PreAct ResNet-18 | DenseNet-161 | Wide ResNet28-10|
| :------ | ---------------: | -----------: |---------------: |
| mixup   |           75.27% |           78.71% |78.86%|
| mwh     |           77.00% |          79.94% |79.94%|




