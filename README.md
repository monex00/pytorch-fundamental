# PyTorch Fundamentals

A structured collection of deep learning projects implemented from scratch in PyTorch. This repository documents my journey learning PyTorch by building classic architectures without relying on pre-built models.

## 🎯 Philosophy

- **From scratch implementation:** No `torchvision.models`, every layer written manually
- **Understanding over results:** Focus on learning PyTorch mechanics, not just accuracy
- **Clean code:** Modular, reusable, and well-documented
- **Progressive complexity:** Start simple (MLP), build up to advanced (Diffusion Models, LoRA)

## 📂 Projects

| #   | Project                       | Dataset  | Key Concepts                           | Status      |
| --- | ----------------------------- | -------- | -------------------------------------- | ----------- |
| 1   | [MLP on MNIST](01_mlp_mnist/) | MNIST    | Training loops, autograd, data loading | ✅ Complete |
| 2   | CNN on CIFAR-10               | CIFAR-10 | Convolutions, data augmentation        | 📋 Planned  |
| 3   | VAE                           | CelebA   | Generative models, latent spaces       | 📋 Planned  |
| 4   | Style Transfer                | Custom   | Feature extraction, optimization       | 📋 Planned  |
| 5   | Transformer                   | Text     | Attention mechanisms, NLP              | 📋 Planned  |

## 🛠️ Setup

### Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/pytorch-fundamentals.git
cd pytorch-fundamentals

# Install dependencies
uv sync

# Navigate to a specific project
cd 01_mlp_mnist
jupyter notebook notebooks/train.ipynb
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

See `pyproject.toml` for full dependencies.

## 📁 Repository Structure

```
pytorch-fundamentals/
├── shared/              # Shared utilities across projects
│   └── utils/
│       └── trainer.py  # Reusable training/validation loops
│
├── 01_mlp_mnist/       # Individual projects
│   ├── models/         # Model architectures
│   ├── notebooks/      # Training experiments
│   └── results/        # Plots and outputs
│
└── 02_cnn_cifar10/     # (future projects follow same structure)
```

## 🧠 What I'm Learning

### Core PyTorch Concepts

- [x] Computational graphs and autograd
- [x] Custom `nn.Module` implementations
- [x] DataLoader and Dataset pipelines
- [x] Training/validation loops
- [x] Device management (CPU/GPU)
- [ ] Learning rate scheduling
- [ ] Model checkpointing
- [ ] Distributed training

### Architectures

- [x] Multi-Layer Perceptrons (MLP)
- [ ] Convolutional Neural Networks (CNN)
- [ ] Variational Autoencoders (VAE)
- [ ] Transformers
- [ ] Diffusion Models

## 🚀 Best Practices Followed

- **Modular code:** Separation of models, training logic, and experiments
- **Reproducibility:** Fixed seeds, documented hyperparameters
- **Version control:** Meaningful commits, clean history
- **Documentation:** README for each project with results and learnings
- **Visualization:** Training curves and qualitative results

## 📚 Resources

- [PyTorch Official Docs](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book](https://www.deeplearningbook.org/)

## 📝 License

MIT License - Free to use for learning purposes.

---

**Author:** [Your Name]**Last Updated:** February 2026

> Each project folder contains its own detailed README with architecture specifics, results, and lessons learned.
