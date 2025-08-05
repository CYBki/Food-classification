# PyTorch ile Gıda Sınıflandırma

Bu proje pizza, biftek ve suşi görüntülerini PyTorch kullanarak sınıflandırmak için hazırlanmıştır. Depo; veri hazırlama, model eğitimi, değerlendirme ve deney takibi gibi uçtan uca bir derin öğrenme iş akışı sunar.

## Öne Çıkanlar

- **Çeşitli modeller:** Özel CNN mimarileri, EfficientNet ve Vision Transformer.
- **Modüler yapı:** Veri yükleme, eğitim döngüsü ve model oluşturma için yeniden kullanılabilir bileşenler.
- **Deney takibi:** TensorBoard ile kayıtlar ve saklanan kontrol noktaları.
- **Dağıtım örnekleri:** Eğitilmiş modelleri dışa aktarma ve kullanma senaryoları.

## Depo Yapısı

```
Food-classification/
├── data/                      # pizza_steak_sushi veri seti
├── Experiment_tracking/       # TensorBoard günlükleri ve kontrol noktaları
├── Model_deployment/          # basit dağıtım demoları
├── PyTorch_Going_Modular/     # modüler eğitim pipeline'ı
├── Transfer__learning/        # transfer öğrenme not defterleri
├── tests/                     # birim testleri
└── README.md
```

## Başlangıç

### Gereksinimler

- Python 3.8+
- Sisteminizde [Git](https://git-scm.com/) ve bir terminal

Gerekli Python paketleri `requirements.txt` dosyasında listelenmiştir.

### Kurulum

```bash
git clone https://github.com/CYBki/Food-classification.git
cd Food-classification

# (İsteğe bağlı) sanal ortam oluşturun
python -m venv .venv
source .venv/bin/activate  # Windows için .venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### Eğitim

Depo, `data/pizza_steak_sushi` dizini altında küçük bir örnek veri seti içerir. Eğer dizin eksikse aşağıdaki komut ile indirilebilir:

```bash
python - <<'PY'
from helper_functions import download_data
download_data(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="pizza_steak_sushi"
)
PY
```

Veri hazır olduğunda eğitim betiği herhangi bir konumdan çalıştırılabilir:

```bash
python PyTorch_Going_Modular/going_modular/train.py
```

TensorBoard günlükleri `Experiment_tracking/runs` dizinine yazılır ve şu şekilde görüntülenebilir:

```bash
tensorboard --logdir ../../Experiment_tracking/runs
```

### Testler

Yardımcı fonksiyonları doğrulamak için birim testlerini çalıştırın:

```bash
pytest
```

## Katkıda Bulunma

Katkılar memnuniyetle karşılanır. Lütfen değişikliklerinizi test edin ve gerekirse dokümantasyonu güncelleyin.

## Lisans

Bu proje [GNU General Public License v3.0](LICENSE) altında dağıtılmaktadır.

## Teşekkür

"PyTorch Going Modular" serisine ve geniş PyTorch topluluğuna teşekkürler.
