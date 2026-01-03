## EfficientFace

![OS](https://img.shields.io/badge/-Linux-grey?logo=linux) ![PyTorch](https://img.shields.io/badge/PyTorch-black?logo=PyTorch) ![License](https://img.shields.io/badge/License-MIT-green) ![Python](https://img.shields.io/badge/python-3.12%2B-blue)

--- 

### Introduction

EfficientFace is a computer vision project focused on **facial keypoint detection** using **transfer learning** with **EfficientNet-B0**. The project explores how a pre-trained convolutional neural network can be adapted to a regression task when only a limited amount of labeled data is available.

In deep learning, training models from scratch is often impractical due to the large datasets and computational resources required. For this reason, **transfer learning** is widely used: a model pre-trained on a large dataset (such as ImageNet) is reused as a feature extractor and adapted to a new task. This approach allows the network to leverage previously learned visual representations, improving convergence speed and generalization.

Although closely related, transfer learning and fine-tuning are not exactly the same. Transfer learning refers to reusing a pre-trained model for a new task, usually by replacing the final layers. Fine-tuning is a specific strategy within transfer learning where some of the pre-trained layers are further trained on the target dataset. In this project, EfficientNet-B0 is used as a backbone and its classification head is replaced with a regression head for facial keypoints, while selected layers are frozen to control the learning process.

The main goal of this project is to gain practical experience with transfer learning while building a complete training and inference pipeline for facial keypoint regression.


---

### Repository Structure

```
EfficientFace/
│
├─ resources/           # Videos, gifs and visual examples
├─ src/                 # Source code
│  ├─ albumentations.py # Custom augmentation functions
│  ├─ utils.py          # Dataset, model definition and training utilities
│  ├─ train.py          # Training pipeline
│  └─ inference.py      # Inference and GIF generation
│
├─ requirements.txt     # Project dependencies
└─ README.md
```