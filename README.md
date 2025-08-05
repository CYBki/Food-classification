# PyTorch ile Gıda Sınıflandırma

Bu depo; pizza, biftek ve suşi görüntülerini sınıflandırmak için PyTorch tabanlı örnekler içerir. Veri hazırlamadan model eğitimine, TensorBoard ile deney takibinden basit dağıtım senaryolarına kadar uçtan uca bir iş akışı sunar.

## Özellikler

* **Çoklu mimariler:** Özel CNN modelleri, EfficientNet ve Vision Transformer.
* **Modüler yapı:** Veri yükleyicileri, eğitim döngüsü ve model bileşenleri yeniden kullanılabilir şekilde düzenlenmiştir.
* **Deney takibi:** TensorBoard günlükleri ve model kontrol noktaları saklanır.
* **Dağıtım örnekleri:** Eğitilen modellerin dışa aktarılması ve kullanılmasına dair basit demolar.
* **Etkileşimli arayüz:** Streamlit tabanlı animasyonlu arayüz sayesinde yüklenen görseller kolayca sınıflandırılabilir.

## Depo Yapısı

```
Food-classification/
├── data/                      # pizza_steak_sushi veri seti (opsiyonel)
├── Experiment_tracking/       # TensorBoard günlükleri ve kontrol noktaları
├── Model_deployment/          # basit dağıtım demoları
├── PyTorch_Going_Modular/     # modüler eğitim pipeline'ı
├── Transfer__learning/        # transfer öğrenme not defterleri
├── tests/                     # birim testleri
└── README.md                  # proje tanıtımı
```

## Kurulum

### 1. Gereksinimler

* Python 3.8+
* [Git](https://git-scm.com/) ve [DVC](https://dvc.org/)
* İsteğe bağlı olarak bir sanal ortam

Gerekli Python paketleri `requirements.txt` dosyasında listelenmiştir.

### 2. Depoyu Klonlama

```bash
# Depoyu klonla
git clone https://github.com/CYBki/Food-classification.git
cd Food-classification

# (Opsiyonel) sanal ortam oluştur
python -m venv .venv
source .venv/bin/activate  # Windows için .venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# DVC ile gerekli model ve veri dosyalarını indir
dvc pull
# Eğer `dvc pull` sırasında "Missing cache files" uyarısı alırsanız,
# modeli içeren makinede `dvc push` komutunu çalıştırarak remote'u güncelleyin.
```

## Veri Seti

Depo, `data/pizza_steak_sushi` dizininde küçük bir örnek veri setiyle çalışacak şekilde tasarlanmıştır. Dizin mevcut değilse aşağıdaki Python komutu ile indirilebilir:

```bash
python download_dataset.py
```

## Model Eğitimi

Veri seti hazır olduktan sonra model eğitimi aşağıdaki komut ile başlatılabilir:

```bash
python PyTorch_Going_Modular/going_modular/train.py
```

Eğitim sırasında oluşan TensorBoard günlükleri `Experiment_tracking/runs` dizinine yazılır. Kayıtları görmek için:

```bash
tensorboard --logdir Experiment_tracking/runs
```

## Web Arayüzü ile Sınıflandırma

Eğitilen modeller Streamlit tabanlı arayüz üzerinden karşılaştırmalı olarak denenebilir. Bir görsel yükledikten sonra TinyVGG,
EfficientNet-B0, EfficientNet-B2 ve ViT-B16 gibi farklı mimariler aynı görüntü üzerinde tahmin yapar. Her bir modelin tahmini ve olasılık değerleri tabloda listelenir, sınıf olasılıkları ise çubuk grafik üzerinde gösterilir.

```bash
streamlit run app.py
```

## Testleri Çalıştırma

Yardımcı fonksiyonların doğru çalıştığından emin olmak için birim testlerini çalıştırın:

```bash
pytest
```

## Katkıda Bulunma

Pull request göndermeden önce lütfen tüm testleri çalıştırın ve dokümantasyonu güncelleyin.

## Lisans

Bu proje [GNU General Public License v3.0](LICENSE) ile lisanslanmıştır.

## Teşekkür

"PyTorch Going Modular" serisi ve geniş PyTorch topluluğuna katkılarından ötürü teşekkür ederiz.
