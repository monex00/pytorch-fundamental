# PyTorch Fundamentals

A structured collection of deep learning projects implemented from scratch in PyTorch. This repository documents a hands-on learning journey through core architectures and training paradigms, from basic MLPs to Neural Style Transfer and Transformers.

## 🎯 Philosophy

- **From scratch implementation:** No `torchvision.models`, every layer written manually
- **Understanding over results:** Focus on learning PyTorch mechanics, not just accuracy
- **Clean code:** Modular, reusable, and well-documented
- **Progressive complexity:** Each project introduces new concepts that build on the previous

## 🎨 Highlights

### Neural Style Transfer — Cat + Van Gogh's Starry Night

![Style Transfer Result](03_style_transfer/results/cat_starry_night.png)

_Content image rendered with Van Gogh's brushstrokes and color palette by optimizing pixel values directly — no training involved, just gradient descent on the image itself._

## 📂 Projects

| #   | Project                                     | Dataset       | Key Concepts                                              | Result                   | Status         |
| --- | ------------------------------------------- | ------------- | --------------------------------------------------------- | ------------------------ | -------------- |
| 1   | [MLP on MNIST](01_mlp_mnist/)               | MNIST         | Autograd, training loops, DataLoaders                     | 98.3% accuracy           | ✅ Complete    |
| 2   | [CNN on CIFAR-10](02_cnn_cifar10/)          | CIFAR-10      | Convolutions, BatchNorm, Dropout                          | 86.3% accuracy           | ✅ Complete    |
| 3   | [Neural Style Transfer](03_style_transfer/) | Custom images | VGG feature extraction, Gram matrices, image optimization | Artistic image synthesis | ✅ Complete    |
| 4   | Transformer for Sentiment                   | IMDb          | Self-attention, positional encoding, NLP                  | TBD                      | 🚧 In Progress |
| 5   | VAE                                         | CelebA        | Generative models, latent spaces                          | TBD                      | 📋 Planned     |

## 🛠️ Setup

### Quick Start

```bash
# Clone the repository
git clone https://github.com/monex00/pytorch-fundamental.git
cd pytorch-fundamental

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+ (nightly build recommended for RTX 40/50 series GPUs)
- CUDA (optional but recommended)

```bash
# For newer GPUs (RTX 40/50 series - Blackwell/Ada architecture)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 📁 Repository Structure

```
pytorch-fundamentals/
│
├── shared/                      # Shared utilities across projects
│   └── utils/
│       └── trainer.py           # Reusable training and validation loops
│
├── 01_mlp_mnist/                # MLP on MNIST
│   ├── models/mlp.py
│   ├── notebooks/train.ipynb
│   └── results/
│
├── 02_cnn_cifar10/              # CNN on CIFAR-10
│   ├── models/
│   │   ├── lenet.py
│   │   └── custom_cnn.py
│   ├── notebooks/train.ipynb
│   └── results/
│
├── 03_style_transfer/           # Neural Style Transfer
│   ├── models/vgg_extractor.py
│   ├── utils/losses.py
│   ├── notebooks/style_transfer.ipynb
│   ├── images/
│   │   ├── content/
│   │   └── style/
│   └── results/
│
└── 04_transformer_sentiment/    # Transformer for Sentiment Analysis (WIP)
```

## 🧠 Learning Progression

### PyTorch Core

- [x] Computational graphs and autograd
- [x] Custom `nn.Module` implementations
- [x] DataLoader and Dataset pipelines
- [x] Training and validation loops
- [x] Device management (CPU/GPU)
- [x] Model evaluation and metrics

### Architectures

- [x] Multi-Layer Perceptrons (MLP)
- [x] Convolutional Neural Networks (CNN)
- [x] VGG-style deep networks (as feature extractor)
- [ ] Transformers and self-attention
- [ ] Variational Autoencoders (VAE)
- [ ] Diffusion Models

### Techniques

- [x] Batch Normalization
- [x] Dropout regularization
- [x] Gram matrix for style representation
- [x] Image optimization (optimizing inputs, not weights)
- [x] Transfer learning with frozen networks
- [ ] Learning rate scheduling
- [ ] Early stopping and model checkpointing
- [ ] Data augmentation

## 🔑 Key Concepts Per Project

### 1. MLP on MNIST

> _"How does PyTorch actually compute gradients?"_

The first project strips everything to the essentials: a simple feedforward network trained on MNIST. Focus is entirely on understanding the training loop, how `loss.backward()` populates gradients, and why `optimizer.zero_grad()` matters.

**Main takeaway:** Autograd builds a computational graph on every forward pass. Backprop traverses it backwards, computing `∂loss/∂param` for every learnable parameter.

### 2. CNN on CIFAR-10

> _"Why do convolutions outperform MLPs on images?"_

Two architectures — the classic LeNet-5 and a modern VGG-style CustomCNN — trained side by side. The 22-point accuracy gap (64% → 86%) makes the impact of BatchNorm, deeper networks, and proper regularization immediately visible.

**Main takeaway:** Architectural choices matter more than hyperparameters. BatchNorm + Dropout + depth = dramatically better generalization.

### 3. Neural Style Transfer

> _"What if we optimize the image instead of the weights?"_

A completely different paradigm: VGG19 is frozen, and we run gradient descent on the pixels themselves. Content is matched by comparing deep feature activations; style is matched by comparing Gram matrices (feature correlations).

**Main takeaway:** Pre-trained networks are versatile feature extractors. Backprop can optimize _any_ differentiable variable — not just weights.

### 4. Transformer (In Progress)

> _"How does a model learn which words to pay attention to?"_

Self-attention mechanism built from scratch, applied to sentiment analysis on IMDb reviews. Multi-head attention, positional encodings, and residual connections all implemented manually.

## 📚 Resources

- [PyTorch Official Docs](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [A Neural Algorithm of Artistic Style (Gatys et al., 2015)](https://arxiv.org/abs/1508.06576)
- [Deep Learning Book (Goodfellow et al.)](https://www.deeplearningbook.org/)

## 📝 License

MIT License - Free to use for learning purposes.

---

**Author:** [monex00](https://github.com/monex00)  
**Last Updated:** May 2026

> Each project folder contains a detailed README with architecture specifics, results, implementation notes, and lessons learned.
