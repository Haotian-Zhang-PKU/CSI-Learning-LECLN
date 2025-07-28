# Wideband Multi-User CSI Learning Using LE-CLN

 This simulation code package is mainly used to reproduce the results of the following paper [1]:

 __[1] H. Zhang, S. Gao, X. Cheng, and L. Yang, "Synesthesia of Machines (SoM)-Enhanced Wideband Multi-User CSI Learning With LiDAR Sensing," IEEE Transactions on Vehicular Technology, early access, doi: 10.1109/TVT.2025.3555130.__

 __If you use this simulation code package in any way, please cite the original paper [1] above.__

 Copyright reserved by the Pervasive Connectivity and Networked Intelligence Laboratory (led by Dr. Xiang Cheng), School of Electronics, Peking University, Beijing 100871, China. 


# Dependencies:
1) Python 3.9.7 

2) Pytorch 1.10.2

3) NVIDIA GPU with a compatible CUDA toolkit (see [NVIDIA website](https://developer.nvidia.com/cuda-toolkit)).

4) Dataset, which can be downloaded at https://pan.baidu.com/s/1zxmSiG3U_zlMzGwymMq1MQ with extraction code my3t.


# Running the code:

1) Ensure that you have the dataset required for LE-CLN.

2) Set the paths of the training and testing datasets in the script "LE_CLN.py".

3) Run LE_CLN.py to get the channel estimation results of LE-CLN. The code first performs LE-CLN training and then obtains inference results.

# Dataset Introduction:
The data interface has been written in LE_CLN.py. You only need to select the corresponding .pth file in the Dataset based on the system configuration (SNR and pilot number), and replace the file path in the code with your own path.

It is noted that there may be some differences in the results of different training processes. 

