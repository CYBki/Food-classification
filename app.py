import time
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd

from PyTorch_Going_Modular.going_modular import model_builder

# Sınıf isimleri
CLASS_NAMES = ["pizza", "steak", "sushi"]

# Modeli yükleme fonksiyonu
@st.cache_resource
def load_model():
    weights_path = Path("PyTorch_Going_Modular/models/05_going_modular_script_mode_tinyvgg_model.pth")
    model = model_builder.TinyVGG(input_shape=3, hidden_units=10, output_shape=len(CLASS_NAMES))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Başlık
st.title("Gıda Sınıflandırma")

# Dosya yükleyici
uploaded_file = st.file_uploader("Bir görüntü yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görüntü", use_column_width=True)

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    with st.spinner("Model tahmin ediyor..."):
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)

        with torch.inference_mode():
            preds = model(image_tensor)
            probs = torch.softmax(preds, dim=1).squeeze()

    pred_label = torch.argmax(probs).item()
    pred_class = CLASS_NAMES[pred_label]
    pred_prob = probs[pred_label].item()

    st.success(f"Tahmin: {pred_class} ({pred_prob:.2%})")
    st.balloons()

    prob_df = pd.DataFrame({"Sınıf": CLASS_NAMES, "Olasılık": probs.tolist()})
    prob_df = prob_df.set_index("Sınıf")
    st.bar_chart(prob_df)
