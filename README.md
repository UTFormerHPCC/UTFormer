# UTFormer: An Ultra-lightweight Transformer Model for Traffic Analysis
    
    
    
## About this project  
This is a development project based on existing work. Following the model architecture, we propose UTFormer, an ultra-lightweight transformer model for rapid traffic analysis.Two technologies are proposed, named dynamic bytes selecting and SGD-based searching for model architecture.Evaluation results on two different datasets (ISCX-VPN-2016、USTC-TFC) and three traffic analysis reveal that compared with other transformer models, UTFormer achieves 196.4-346.7X reduction on computation.

 
## Prerequisites
### Install python packages

    click==8.1.7
    numpy~=1.24.4
    pandas==2.0.3
    joblib==1.3.2
    scapy==2.4.5
    scipy==1.10.1
    ipykernel==6.26.0
    torch==2.1.1
    torcheval==0.0.7
    scikit-learn==1.3.2
    tqdm==4.66.1
    matplotlib==3.7.4
    wandb==0.16.0
    torchinfo==1.8.0
    packaging==23.2
    torchvision==0.16.1

### Download datasets
The validation of our UTFormer model is implemented on ISCX-VPN-2016 (ISCX) [25] dataset and USTC-TFC (USTC) dataset with three different traffic analysis tasks. ISCX dataset contains of 6 categories of encrypted traffic from 17 different applications. We train UTFormer for encrypted traffic classification on ISXC. USTC dataset contains traffic from 19 real-world applications with 9 different services. Traffic from 6 common malware applications is also collected by USTC dataset. UTFormer is trained on USTC for service traffic classification and malware traffic detection respectively.
- [ISCX-VPN-2016]("https://www.unb.ca/cic/datasets/vpn.html")
- [USTC-TFC]("https://github.com/yungshenglu/USTC-TFC2016")
 
# Directory description
    ├── readme.md           
    
    ├── data           //pickle files for train, test, and validation  
    
    ├── result         // expreriment results folder
    
    ├── models         // trained models

    ├── mtt_1.py         

    ├── mtt_block.py         

    ├── train.py         

    ├── preprocess.py         
    
    └── dataset.py           
 

 # Lisence
    Distributed under the MIT License. 
 
# Version Update
###### v1.0.0: 
    1. upload source code
    2. upload trained models
 
 
