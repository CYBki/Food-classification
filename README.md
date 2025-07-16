# 🍕 Food Classification with PyTorch

A comprehensive PyTorch-based deep learning project for classifying food images (pizza, steak, and sushi) using modern computer vision techniques, transfer learning, and modular architecture.

## 📋 Project Overview

This repository demonstrates practical implementation of PyTorch for food image classification, featuring:

- **Deep Learning Models**: Custom CNN architectures and pre-trained models
- **Transfer Learning**: Leveraging EfficientNet and Vision Transformer (ViT) models
- **Experiment Tracking**: TensorBoard integration for monitoring training progress
- **Modular Design**: Clean, reusable PyTorch code structure
- **Educational Focus**: Based on PyTorch Going Modular tutorials

The project classifies food images into three categories: **Pizza** 🍕, **Steak** 🥩, and **Sushi** 🍣.

## 📁 Repository Structure

```
Food-classification/
├── Transfer__learning/              # Transfer learning experiments
│   ├── PyTorch_transfer_learning.ipynb
│   ├── data/pizza_steak_sushi/
│   └── helper_functions.py
├── Experiment_tracking/             # TensorBoard experiment tracking
│   ├── PyTorch Experiment Tracking.ipynb
│   ├── models/                     # Saved model checkpoints
│   └── runs/                       # TensorBoard log files
├── PyTorch_Going_Modular/          # Modular PyTorch architecture
│   ├── going_modular/
│   │   ├── data_setup.py          # Data loading utilities
│   │   ├── engine.py              # Training/testing loops
│   │   ├── model_builder.py       # Model architectures
│   │   ├── predictions.py         # Prediction utilities
│   │   ├── train.py               # Training script
│   │   └── utils.py               # Helper functions
│   └── 05_pytorch_going_modular_script_mode.ipynb
├── PyTorch_paper_replicating/      # Research paper implementations
├── Model_deployment/               # Model deployment examples
├── data/                          # Training and test datasets
│   └── pizza_steak_sushi/
│       ├── train/                 # Training images
│       └── test/                  # Test images
├── Custom_dataset.ipynb           # Custom dataset creation
├── helper_functions.py            # Utility functions
└── LICENSE                        # GPL-3.0 License
```

## ✨ Key Features

- **🔄 Transfer Learning**: Pre-trained EfficientNet and Vision Transformer models
- **📊 Experiment Tracking**: TensorBoard integration for loss and accuracy monitoring
- **🧩 Modular Architecture**: Separated concerns with dedicated modules for data, models, and training
- **📈 Performance Comparison**: Multiple model architectures and training strategies
- **🎯 Food Classification**: Specialized for pizza, steak, and sushi recognition
- **📚 Educational Value**: Step-by-step learning progression with detailed notebooks

## 🛠️ Technologies Used

- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[torchvision](https://pytorch.org/vision/)** - Computer vision library
- **[TensorBoard](https://www.tensorflow.org/tensorboard)** - Experiment tracking and visualization
- **Python 3.7+** - Programming language
- **Jupyter Notebooks** - Interactive development environment
- **matplotlib** - Data visualization
- **PIL (Pillow)** - Image processing

## 🚀 Getting Started

### Prerequisites

- **Python 3.7+** 
- **Git** (for cloning the repository)
- **Internet connection** (for downloading dependencies and dataset)

### Quick Setup

Follow these steps to get the project running on your machine:

#### 1. **Clone the repository**
```bash
git clone https://github.com/CYBki/Food-classification.git
cd Food-classification
```

#### 2. **Install dependencies**
```bash
# Install PyTorch and related libraries
pip install torch torchvision torchaudio

# Install additional dependencies  
pip install matplotlib pillow tensorboard jupyter tqdm requests
```

#### 3. **Download the dataset**
```bash
# Run the automated dataset download script
python download_data.py
```
This script will:
- ✅ Download the pizza_steak_sushi dataset (if not already present)
- ✅ Extract it to the correct directory structure
- ✅ Verify the setup was successful

#### 4. **Train the model**
```bash
# Navigate to the training script directory
cd PyTorch_Going_Modular/going_modular

# Start training with informative output
python train.py
```

#### 5. **Verify installation (Optional)**
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Alternative Training Locations

The repository contains multiple copies of the training script for different experiments:

```bash
# Main modular training (recommended)
cd PyTorch_Going_Modular/going_modular && python train.py

# Transfer learning experiments  
cd Transfer__learning/PyTorch_Going_Modular/going_modular && python train.py

# Experiment tracking version
cd Experiment_tracking/PyTorch_Going_Modular/going_modular && python train.py

# Paper replication version
cd PyTorch_paper_replicating/PyTorch_Going_Modular/going_modular && python train.py
```

### Interactive Exploration

For Jupyter notebook exploration:
```bash
jupyter notebook
```

For TensorBoard monitoring:
```bash
tensorboard --logdir=Experiment_tracking/runs
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### ❌ `FileNotFoundError: [Errno 2] No such file or directory: 'data/pizza_steak_sushi/train'`

**Solution:** Run the dataset download script first:
```bash
python download_data.py
```

#### ❌ `ModuleNotFoundError: No module named 'torch'`

**Solution:** Install PyTorch:
```bash
pip install torch torchvision torchaudio
```

#### ❌ `ModuleNotFoundError: No module named 'tqdm'`

**Solution:** Install missing dependencies:
```bash
pip install tqdm requests
```

#### ❌ Internet connection issues during dataset download

**Solution:** Check your internet connection and try again:
```bash
# Remove incomplete download and retry
rm -rf data/pizza_steak_sushi
python download_data.py
```

#### ❌ Permission errors when saving models

**Solution:** Ensure you have write permissions in the directory:
```bash
chmod +w models/
# Or run from a directory where you have write permissions
```

#### ❌ CUDA out of memory errors

**Solution:** Reduce batch size in train.py:
```python
BATCH_SIZE = 16  # or smaller
```

### Getting Help

If you encounter other issues:

1. **Check the [Issues](https://github.com/CYBki/Food-classification/issues)** page for known problems
2. **Create a new issue** with:
   - Your operating system
   - Python version (`python --version`)
   - PyTorch version
   - Complete error message
   - Steps you've already tried

## 📊 Training Customization

You can customize the training process by modifying the hyperparameters in `train.py`:

```python
# Setup hyperparameters
NUM_EPOCHS = 5          # Number of training epochs
BATCH_SIZE = 32         # Batch size for training
HIDDEN_UNITS = 10       # Hidden units in the TinyVGG model
LEARNING_RATE = 0.001   # Learning rate for the optimizer
```

### Recommended Settings

For **faster training** (CPU):
```python
NUM_EPOCHS = 3
BATCH_SIZE = 16
```

For **better accuracy** (if you have GPU):
```python
NUM_EPOCHS = 10
BATCH_SIZE = 64
HIDDEN_UNITS = 20
```

For **experimentation**:
```python
LEARNING_RATE = 0.01    # Higher learning rate
HIDDEN_UNITS = 50       # More complex model
```

## 📁 Updated Project Structure

```
Food-classification/
├── 📜 download_data.py              # 🆕 Automated dataset download script
├── 📂 data/                         # Training and test datasets
│   └── pizza_steak_sushi/           # Food classification dataset
│       ├── train/                   # Training images (pizza, steak, sushi)
│       └── test/                    # Test images (pizza, steak, sushi)
├── 📂 PyTorch_Going_Modular/        # Main modular PyTorch implementation
│   └── going_modular/
│       ├── train.py                 # 🔄 Updated training script with better paths & output
│       ├── data_setup.py            # Data loading utilities
│       ├── engine.py                # Training/testing loops  
│       ├── model_builder.py         # Model architectures
│       ├── predictions.py           # Prediction utilities
│       └── utils.py                 # Helper functions
├── 📂 Transfer__learning/           # Transfer learning experiments  
├── 📂 Experiment_tracking/          # TensorBoard experiment tracking
├── 📂 PyTorch_paper_replicating/    # Research paper implementations
├── 📂 Model_deployment/             # Model deployment examples
├── 📝 Custom_dataset.ipynb          # Custom dataset creation
├── 📝 helper_functions.py           # Utility functions
├── 📖 README.md                     # 🔄 Updated documentation
└── 📄 LICENSE                       # GPL-3.0 License
```

## 🍽️ Food Classification Details

### Dataset
- **Classes**: 3 food categories (Pizza, Steak, Sushi)
- **Structure**: Organized in train/test split
- **Format**: RGB images in various resolutions
- **Preprocessing**: Resize to 64x64 or 224x224 depending on model

### Model Architectures

#### 1. Custom TinyVGG
```python
model = TinyVGG(
    input_shape=3,
    hidden_units=10,
    output_shape=3  # pizza, steak, sushi
)
```

#### 2. Transfer Learning with EfficientNet
```python
from torchvision import models
model = models.efficientnet_b0(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 3)
```

#### 3. Vision Transformer (ViT)
```python
model = models.vit_b_16(pretrained=True)
model.heads = nn.Linear(model.heads.head.in_features, 3)
```

## 🎓 Learning Objectives

This project demonstrates:

- **PyTorch Fundamentals**: Tensors, datasets, dataloaders, and neural networks
- **Computer Vision**: Image preprocessing, augmentation, and CNN architectures
- **Transfer Learning**: Fine-tuning pre-trained models for specific tasks
- **Model Training**: Training loops, loss functions, and optimization
- **Experiment Tracking**: Monitoring and comparing different approaches
- **Code Organization**: Modular programming and best practices
- **Model Deployment**: Saving and loading trained models

## 💻 Usage Examples

### Training a Custom Model

```python
import torch
from torchvision import transforms
from going_modular import data_setup, model_builder, engine

# Setup data
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir="data/pizza_steak_sushi/train",
    test_dir="data/pizza_steak_sushi/test",
    transform=transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ]),
    batch_size=32
)

# Create model
model = model_builder.TinyVGG(input_shape=3, hidden_units=10, output_shape=3)

# Train model
results = engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=torch.optim.Adam(model.parameters()),
    loss_fn=torch.nn.CrossEntropyLoss(),
    epochs=5,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

### Making Predictions

```python
from going_modular import predictions
import torch

# Load trained model
model = torch.load("models/food_classifier.pth")

# Make prediction and plot image
predictions.pred_and_plot_image(
    model=model,
    image_path="path/to/food/image.jpg",
    class_names=["pizza", "steak", "sushi"],
    image_size=(224, 224)
)
```

### Command Line Training

```bash
# Train with the default configuration
cd PyTorch_Going_Modular/going_modular
python train.py
```

The training script uses predefined hyperparameters that can be modified directly in the `train.py` file:
- Epochs: 5
- Batch size: 32
- Learning rate: 0.001
- Hidden units: 10

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests if applicable**
5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Include example usage in notebooks
- Update documentation as needed
- Test your changes thoroughly

## 📄 License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author & Acknowledgments

### Author
- **CYBki** - *Project Creator and Maintainer*

### Acknowledgments

- **[PyTorch Team](https://pytorch.org/)** - For the amazing deep learning framework
- **[PyTorch Tutorials](https://pytorch.org/tutorials/)** - Educational foundation and inspiration
- **[Daniel Bourke](https://github.com/mrdbourke)** - PyTorch Going Modular tutorial series
- **torchvision Contributors** - Pre-trained models and datasets
- **Open Source Community** - For continuous learning and sharing

### Learning Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Computer Vision with PyTorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

⭐ **Star this repository if you found it helpful!**

🐛 **Found a bug?** Please open an issue with detailed information.

📧 **Questions?** Feel free to reach out through GitHub issues.