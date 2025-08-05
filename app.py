import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import pandas as pd
from pickle import UnpicklingError
import subprocess
import shutil

from PyTorch_Going_Modular.going_modular import model_builder


# Repository root (current file lives at project root)
REPO_ROOT = Path(__file__).resolve().parent

# Available class names for predictions
CLASS_NAMES = ["pizza", "steak", "sushi"]


def create_effnet_b0(num_classes: int) -> torch.nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def create_effnet_b2(num_classes: int) -> torch.nn.Module:
    model = models.efficientnet_b2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def create_vit_b16(num_classes: int) -> torch.nn.Module:
    model = models.vit_b_16(weights=None)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    return model


MODEL_INFO = {
    "TinyVGG": {
        "path": Path("PyTorch_Going_Modular/models/05_going_modular_script_mode_tinyvgg_model.pth"),
        "builder": lambda: model_builder.TinyVGG(input_shape=3, hidden_units=10, output_shape=len(CLASS_NAMES)),
        "transform": transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()]),
    },
    "EfficientNet-B0": {
        "path": Path("Experiment_tracking/models/07_effnetb0_data_20_percent_10_epochs.pth"),
        "builder": lambda: create_effnet_b0(len(CLASS_NAMES)),
        "transform": transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
    },
    "EfficientNet-B2": {
        "path": Path("Experiment_tracking/models/07_effnetb2_data_20_percent_10_epochs.pth"),
        "builder": lambda: create_effnet_b2(len(CLASS_NAMES)),
        "transform": transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
    },
    "ViT-B16": {
        "path": Path("PyTorch_paper_replicating/models/08_pretrained_vit_feature_extractor_pizza_steak_sushi.pth"),
        "builder": lambda: create_vit_b16(len(CLASS_NAMES)),
        "transform": transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
    },
}


@st.cache_resource
def load_model(name: str) -> torch.nn.Module:
    info = MODEL_INFO[name]
    path = info["path"]

    # If a leftover Git LFS pointer file is present, remove it so DVC can pull the real weights
    if path.exists():
        try:
            if path.read_text().startswith("version https://git-lfs.github.com/spec/v1"):
                path.unlink()
        except UnicodeDecodeError:
            pass

    if not path.exists():
        dvc_file = path.with_suffix(path.suffix + ".dvc")
        if dvc_file.exists():
            if shutil.which("dvc") is None:
                raise FileNotFoundError(
                    "DVC command not found. Install DVC and run 'dvc pull' to download the model."
                )
            try:
                subprocess.run(
                    ["dvc", "pull", str(dvc_file.relative_to(REPO_ROOT))],
                    cwd=REPO_ROOT,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as pull_error:
                raise FileNotFoundError(
                    f"Model file not found: {path}. Run 'dvc pull' to download the model. "
                    f"Original error: {pull_error.stderr.decode().strip()}"
                ) from pull_error
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}. Run 'dvc pull' to download the model."
            )

    model = info["builder"]()
    try:
        state_dict = torch.load(path, map_location="cpu")
    except (UnpicklingError, RuntimeError, EOFError) as e:
        raise RuntimeError(
            f"Failed to load PyTorch model from {path}. "
            f"The file might be corrupted or not a valid PyTorch state dict. "
            f"Original error: {str(e)}"
        ) from e

    model.load_state_dict(state_dict)
    model.eval()
    return model


st.title("Gıda Sınıflandırma Karşılaştırması")

selected_models = st.multiselect(
    "Modelleri seçin",
    list(MODEL_INFO.keys()),
    default=list(MODEL_INFO.keys()),
)

uploaded_file = st.file_uploader(
    "Bir görüntü yükleyin", type=["jpg", "jpeg", "png"]
)

if uploaded_file and selected_models:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görüntü", use_column_width=True)

    results = {}
    summary_rows = []

    for model_name in selected_models:
        try:
            model = load_model(model_name)
        except Exception as e:
            st.error(f"Error loading {model_name}: {str(e)}")
            continue

        transform = MODEL_INFO[model_name]["transform"]
        image_tensor = transform(image).unsqueeze(0)

        with st.spinner(f"{model_name} tahmin ediyor..."):
            with torch.inference_mode():
                preds = model(image_tensor)
                probs = torch.softmax(preds, dim=1).squeeze()

        pred_idx = int(torch.argmax(probs))
        summary_rows.append(
            {
                "Model": model_name,
                "Tahmin": CLASS_NAMES[pred_idx],
                "Olasılık": float(probs[pred_idx]),
            }
        )
        results[model_name] = probs.tolist()

    if summary_rows:  # Only show results if we have any successful predictions
        st.subheader("Tahmin Karşılaştırması")
        st.table(pd.DataFrame(summary_rows))

        chart_df = pd.DataFrame(results, index=CLASS_NAMES)
        st.bar_chart(chart_df)
