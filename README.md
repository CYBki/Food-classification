# PyTorch ile Gıda Sınıflandırma

Bu depo; pizza, biftek ve suşi görüntülerini sınıflandırmak için PyTorch tabanlı örnekler içerir. Veri hazırlamadan model eğitimine, TensorBoard ile deney takibinden basit dağıtım senaryolarına kadar uçtan uca bir iş akışı sunar.

## Özellikler
- **Çoklu mimariler:** Özel CNN modelleri, EfficientNet ve Vision Transformer.
- **Modüler yapı:** Veri yükleyicileri, eğitim döngüsü ve model bileşenleri yeniden kullanılabilir şekilde düzenlenmiştir.
- **Deney takibi:** TensorBoard günlükleri ve model kontrol noktaları saklanır.
- **Dağıtım örnekleri:** Eğitilen modellerin dışa aktarılması ve kullanılmasına dair basit demolar.

## Depo Yapısı
```
Food-classification/
├── data/                      # pizza_steak_sushi veri seti (opsiyonel)
├── Experiment_tracking/       # TensorBoard günlükleri ve kontrol noktaları
├── Model_deployment/          # basit dağıtım demoları
├── PyTorch_Going_Modular/     # modüler eğitim pipeline'ı
├── Transfer__learning/        # transfer öğrenme not defterleri
├── tests/                     # birim testleri
└── README.md
```

## Kurulum
### 1. Gereksinimler
- Python 3.8+
- [Git](https://git-scm.com/) ve [Git LFS](https://git-lfs.com/)
- İsteğe bağlı olarak bir sanal ortam

Gerekli Python paketleri `requirements.txt` dosyasında listelenmiştir.

### 2. Depoyu Klonlama
```bash
# Git LFS'i etkinleştir
git lfs install

# Depoyu klonla (LFS dosyalarını klonlama sırasında atla)
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/CYBki/Food-classification.git
cd Food-classification

# Eksik LFS dosyalarını indirmeyi dene
git lfs fetch --all
git lfs pull

# (Opsiyonel) sanal ortam oluştur
python -m venv .venv
source .venv/bin/activate  # Windows için .venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# Python 3.13 ve üzeri için TensorBoard'un gerektirdiği `imghdr` modülünü ayrıca yükleyin
# (bu adım `ModuleNotFoundError: imghdr` hatasını giderir)
# pip install imghdr
```

## Veri Seti
Depo, `data/pizza_steak_sushi` dizininde küçük bir örnek veri setiyle çalışacak şekilde tasarlanmıştır. Dizin mevcut değilse aşağıdaki Python komutu ile indirilebilir:
```bash
python - <<'PY'
from helper_functions import download_data

download_data(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="pizza_steak_sushi"
)
PY
```

## Model Eğitimi
Veri seti hazır olduktan sonra model eğitimi herhangi bir konumdan aşağıdaki komut ile başlatılabilir:
```bash
python PyTorch_Going_Modular/going_modular/train.py
```
Eğitim sırasında oluşan TensorBoard günlükleri `Experiment_tracking/runs` dizinine yazılır. Kayıtları görmek için:
```bash
tensorboard --logdir Experiment_tracking/runs
```

## Web Arayüzü ile Sınıflandırma
Eğitilen modeli Streamlit tabanlı basit bir arayüz üzerinden denemek için:
```bash
streamlit run app.py
```
Komut çalıştıktan sonra açılan sayfadan bir görsel yükleyip model tahminini görebilirsiniz.
Model yalnızca pizza, biftek ve suşi sınıfları için eğitimlidir; farklı yiyecekler yanlış sonuç verebilir.

## Testleri Çalıştırma
Yardımcı fonksiyonların doğru çalıştığından emin olmak için birim testlerini çalıştırın:
```bash
pytest
```
`pytest` komutu bulunamıyorsa bağımlılıkların yüklendiğinden ve doğru Python ortamının aktif olduğundan emin olun.

## Katkıda Bulunma
Pull request göndermeden önce lütfen tüm testleri çalıştırın ve gerekliyse dokümantasyonu güncelleyin.

## Lisans
Bu proje [GNU General Public License v3.0](LICENSE) ile lisanslanmıştır.

## Teşekkür
"PyTorch Going Modular" serisi ve geniş PyTorch topluluğuna katkılarından ötürü teşekkür ederiz.
