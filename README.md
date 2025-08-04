# Food Classification with PyTorch

Classify images of pizza, steak and sushi using PyTorch.  The repository demonstrates a minimal yet complete deep learning workflow including data preparation, model training, evaluation and experiment tracking.

## Highlights

* **Multiple models:** custom CNNs, EfficientNet and Vision Transformer.
* **Modular code:** reusable components for data loading, training loops and model building.
* **Experiment tracking:** TensorBoard runs and saved checkpoints.
* **Deployment examples:** scripts and notebooks that show how to export and use trained models.

## Repository layout

```
Food-classification/
├── data/                      # pizza_steak_sushi dataset
├── Experiment_tracking/       # TensorBoard logs and model checkpoints
├── Model_deployment/          # simple deployment demos
├── PyTorch_Going_Modular/     # modular training pipeline
├── Transfer__learning/        # transfer learning notebooks
├── tests/                     # unit tests
└── README.md
```

## Getting started

### Prerequisites

* Python 3.8+
* [PyTorch](https://pytorch.org/)
* torchvision
* matplotlib
* Pillow
* tensorboard

### Installation

```bash
git clone https://github.com/CYBki/Food-classification.git
cd Food-classification
pip install torch torchvision torchaudio
pip install matplotlib pillow tensorboard
```

### Training

```bash
cd PyTorch_Going_Modular/going_modular
python train.py
```

TensorBoard logs are written to `Experiment_tracking/runs` and can be viewed with:

```bash
tensorboard --logdir ../../Experiment_tracking/runs
```

### Tests

Run the unit tests to verify the helper functions:

```bash
pytest
```

## Contributing

Pull requests are welcome.  Please test your changes and update the documentation when necessary.

## License

Released under the [GNU General Public License v3.0](LICENSE).

## Acknowledgements

Based on the "PyTorch Going Modular" series by [Daniel Bourke](https://github.com/mrdbourke) and the wider PyTorch community.
