"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
from pathlib import Path

import torch

from torchvision import transforms
from timeit import default_timer as timer

import data_setup, engine, model_builder, utils


# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories using pathlib.Path for cross-platform compatibility
# Get the repository root directory (2 levels up from this script)
repo_root = Path(__file__).resolve().parent.parent.parent
data_dir = repo_root / "data" / "pizza_steak_sushi"
train_dir = data_dir / "train"
test_dir = data_dir / "test"

print("=" * 60)
print("🍕 Food Classification Training with PyTorch")
print("=" * 60)

# Verify data directories exist
print(f"📂 Data directory: {data_dir}")
if not train_dir.exists():
    print(f"❌ Training directory not found: {train_dir}")
    print("💡 Please run the dataset download script first:")
    print("   python download_data.py")
    exit(1)

if not test_dir.exists():
    print(f"❌ Test directory not found: {test_dir}")
    print("💡 Please run the dataset download script first:")
    print("   python download_data.py")
    exit(1)

print(f"✅ Training directory: {train_dir}")
print(f"✅ Test directory: {test_dir}")

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Using device: {device}")

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])
print(f"🔄 Image transforms: Resize(64, 64) + ToTensor()")

print(f"\n📋 Training Configuration:")
print(f"   • Epochs: {NUM_EPOCHS}")
print(f"   • Batch size: {BATCH_SIZE}")
print(f"   • Learning rate: {LEARNING_RATE}")
print(f"   • Hidden units: {HIDDEN_UNITS}")

print(f"\n🔄 Creating data loaders...")

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=str(train_dir),  # Convert Path to string for compatibility
    test_dir=str(test_dir),
    transform=data_transform,
    batch_size=BATCH_SIZE
)

print(f"✅ Data loaders created successfully!")
print(f"📊 Found {len(class_names)} classes: {class_names}")
print(f"🚂 Training batches: {len(train_dataloader)}")
print(f"🧪 Test batches: {len(test_dataloader)}")

print(f"\n🏗️  Creating TinyVGG model...")

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

print(f"✅ Model created and moved to {device}")

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

print(f"📉 Loss function: CrossEntropyLoss")
print(f"⚡ Optimizer: Adam (lr={LEARNING_RATE})")

if __name__ == "__main__":
    print(f"\n🚀 Starting training...")
    print("=" * 60)
    
    # Start timing
    start_time = timer()
    
    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS,
                 device=device)

    # End timing
    end_time = timer()
    total_time = end_time - start_time
    
    print("=" * 60)
    print(f"✅ Training completed!")
    print(f"⏱️  Total training time: {total_time:.3f} seconds")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save the model with help from utils.py
    model_save_path = "models/05_going_modular_script_mode_tinyvgg_model.pth"
    utils.save_model(model=model,
                     target_dir="models",
                     model_name="05_going_modular_script_mode_tinyvgg_model.pth")
    
    print(f"💾 Model saved to: {model_save_path}")
    print("🎉 Training pipeline completed successfully!")
    print("\n💡 Next steps:")
    print("   • Use the saved model for predictions")
    print("   • Experiment with different hyperparameters")
    print("   • Try transfer learning with pre-trained models")
    print("=" * 60)
