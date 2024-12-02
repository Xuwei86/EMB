Efficient Mamba (EMB)
A Lightweight and Powerful Framework for Visual Tasks

Efficient Mamba (EMB) is a novel framework designed to address the limitations of Mamba models in capturing long-range dependencies and enhancing generalization across diverse computer vision tasks. EMB combines Convolutional Neural Networks (CNNs), Transformers, and State Space Models (SSMs) to achieve state-of-the-art performance with competitive efficiency.

This repository provides the key modules, training scripts, and pretrained weights for Efficient Mamba.

Key Features
TransSSM Module:

Enhances global feature representation using feature flipping and channel shuffling.
Integrates Dual Pooling Attention (DPA) to improve stability and channel-wise attention.
Window Spatial Attention (WSA) Module:

Combines windowed multi-head self-attention with depthwise convolutions for precise local feature modeling.
MultiFusion Block (MFB) and Spatial-Channel Fusion Block (SCFB):

Efficiently integrates local and global feature representations, balancing complexity and accuracy.
Resource Efficiency:

Achieves high accuracy on ImageNet-1k, MS COCO, and ADE20K with minimal parameters and FLOPs.
Model	Parameters	FLOPs	Top-1 Accuracy (ImageNet-1k)
EMB-S	5.9M	1.5G	78.9%
EMB-T	2.5M	0.6G	76.3%
EMB-N	1.4M	0.3G	73.5%

Modules Overview
TransSSM Module
A core module leveraging feature flipping and channel shuffling to enhance global feature modeling. It mitigates the limitations of traditional SSMs.

WSA (Window Spatial Attention)
Focuses on capturing local features with a windowed self-attention mechanism, combined with depthwise convolutions for efficiency.

MultiFusion Block (MFB)
Combines the strengths of TransSSM and WSA, enabling balanced local and global feature extraction.
