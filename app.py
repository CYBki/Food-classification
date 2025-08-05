import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

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
st.write(
    "Bu demo model yalnızca **pizza**, **biftek** ve **suşi** sınıflarını içerir. "
    "Diğer yiyecekler yanlış sınıflandırılabilir."
)

# Dosya yükleyici
uploaded_file = st.file_uploader("Bir görüntü yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görüntü", use_column_width=True)

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    with torch.inference_mode():
        preds = model(image_tensor)
        probs = torch.softmax(preds, dim=1)
        pred_label = probs.argmax(dim=1).item()
        pred_class = CLASS_NAMES[pred_label]
        pred_prob = probs[0][pred_label].item()

    st.write(f"Tahmin: {pred_class} ({pred_prob:.2f})")
    st.write("Sınıf olasılıkları:")
    for class_name, prob in zip(CLASS_NAMES, probs[0]):
        st.write(f"- {class_name}: {prob:.2f}")
