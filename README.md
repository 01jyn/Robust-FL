# A Balanced coarse-to-fine federated learning framework for noisy heterogeneous clients
                                           Framework
 ![image](https://github.com/01jyn/Robust-FL/assets/95575818/1e4d4a8d-4c86-4366-83fe-a8bdf1643e88)

# [Requirements](#contents) 
    Python 3.10.8  
  
    Pytorch 1.12.1
  
# [Preparation](#contents)  
**Dataset**

  Our experiments are conducted on two datasets, Cifar10 and Cifar100. We set public dataset on the server 
  as a subset of Cifar100, and randomly divide Cifar10 to different clients as private datasets.

  1. Dataset used: [CIFAR-10、CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html)

  2. Download the CIFAR-10、CIFAR-100 datasets and extract them to Dataset/

Note: Data will be processed in init_data.py

# [Training](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:
```
cd Robust_FL/HHF
```
# [Change configs](#contents)  
  **Change the parameters in Robust_FL/HHF/HHF.py**

1. ```Dataset_dir``` should be the file path of the training dataset.
2.  ```Network_dir``` will store the models. 
3.  ```Logs_dir```will store the logs
   
# [Load pretrained model](#contents)

   ```python pretrain.py --Pytorch -- ./Robust_FL/Network/```

 # [Train the model](#contents)
 ```python HHF/HHF.py first --Pytorch -- ./Robust_FL/HHF/```

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
```
# [References](#contents)
- Article:[Robust Federated Learning with Noisy and Heterogeneous Client](CVPR 2022 Open Access Repository)
  Xiuwen Fang, Mang Ye CVPR 2022
- Code:https://github.com/FangXiuwen/Robust_FL
