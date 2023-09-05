# Robust federated learning based on self-paced learning

This repository provides resources for the following paper:

> [**Robust federated learning based on self-paced learning**]

# [Dataset](#contents)

Our experiments are conducted on two datasets, Cifar10 and Cifar100. We set public dataset on the server as a subset of Cifar100, and randomly divide Cifar10 to different clients as private datasets.

Dataset used: [CIFAR-10、CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html)

Note: Data will be processed in init_data.py

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash
# init public data and local data
python Dataset/init_data.py
# pretrain local models
python Network/pretrain.py
# RHFL
python HHF/HHF.py
```

# [Script and Sample Code](#contents)

```bash
├── Robust_FL
    ├── Dataset
        ├── cifar.py
        ├── init_dataset.py
        ├── utils.py
    ├── Network
        ├── Models_Def
            ├── mobilnet_v2.py
            ├── resnet.py
            ├── shufflenet.py
        ├── pretrain.py
    ├── regularization
        ├── regularizer.py
    ├── HHF
        ├── HHF.py       
    ├── loss.py
        ├── weight_loss.py
    ├── README.md

