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
- Sisteminizde [Git](https://git-scm.com/), [Git LFS](https://git-lfs.com/) ve bir terminal

Gerekli Python paketleri `requirements.txt` dosyasında listelenmiştir.

### Kurulum

Bu depo bazı büyük dosyalar için Git LFS kullanır. LFS yapılandırılmadıysa veya klonlama
sırasında LFS nesneleri eksik uyarısı alırsanız aşağıdaki adımları izleyin:

```bash
# Git LFS'i etkinleştirin
git lfs install

# LFS dosyalarını indirme işlemini klonlama sırasında atlayın
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/CYBki/Food-classification.git
cd Food-classification

# Eksik dosyaları indirmeyi deneyin (bazıları sunucuda olmayabilir)
git lfs fetch --all

# Bu adım "Object does not exist on the server" hatası döndürebilir; ilgili
# büyük dosyalar LFS sunucusunda mevcut değildir ve çoğu kullanım senaryosu için
# gerekli değildir. Hata alsanız bile devam edebilirsiniz.
git lfs pull || true

# (İsteğe bağlı) sanal ortam oluşturun
python -m venv .venv
source .venv/bin/activate  # Windows için .venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
# Bu adımı atlamak `numpy` veya `packaging` gibi modüllerin eksik olmasına
# neden olur.
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
