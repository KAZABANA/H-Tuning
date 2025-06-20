# A Pytorch implementation of our paper H-Tuning: Toward Low-Cost and Efficient ECG-based Cardiovascular Disease Detection with Pre-Trained Models
# [ICML 2025](https://icml.cc/virtual/2025/poster/45284)
# Preliminaries
* Four downstream datasets: The Chapman-Shaoxing dataset, the PTB-XL dataset, the Ningbo dataset, and the G12EC dataset.
* quick download the four downsteam datasets: wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/
* The pre-trained backbones are available on [Hugging Face](https://huggingface.co/KAZABANA/Foundation-Models-for-ECG-classification/tree/main).
* Requirements: you can use (pip install -r requirements.txt) to install all the related packages used in our paper.
# H-Tuning Fine-Tuning and Evaluation on the Downsteam Datasets.
* In this repo, we provide the backbones pre-trained by our team, so you can directly fine-tune them on the downstream datasets using the proposed H-Tuning method.
* First, you can use the function 'dataset_organize' in the datacollection.py to preprocess the four downstream datasets.
* Then, in the command window, you can the command below to reproduce the results of H-Tuning. The saved output file presents the performance of the algorithm.
```
nohup python -u main.py --model_config 'medium' --finetune_label_ratio 0.10 --task 'within' --r 16 --seed 18 --device "cuda:3" > training_ecg_seed18.log 2>&1 &
```
# Citation
If you find our work helps your research, please kindly consider citing our paper in your publications.
@inproceedings{
zhou2025htuning,
title={H-Tuning: Toward Low-Cost and Efficient {ECG}-based Cardiovascular Disease Detection with Pre-Trained Models},
author={Rushuang Zhou and Yuanting Zhang and Yining Dong},
booktitle={Forty-second International Conference on Machine Learning},
year={2025}
}

