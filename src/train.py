from ultralytics import YOLO
import os

def train_model(data_path, epochs=20):
    # data_path içinde data.yaml dosyasını bulmamız lazım
    yaml_path = os.path.join(data_path, "data.yaml")
    
    print(f"🔥 Eğitim başlıyor... Konfigürasyon: {yaml_path}")
    
    # YOLOv8 nano modelini yükle (en hafifi)
    model = YOLO('yolov8n.pt') 

    # Eğitimi başlat
    # device='mps' Mac için, '0' Nvidia GPU için, 'cpu' işlemci için.
    # Otomatik bırakırsan kütüphane en iyisini seçer.
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        name='akne_modeli'  # Çıktı klasörünün adı
    )
    
    print("✅ Eğitim tamamlandı!")
    return model

if __name__ == "__main__":
    # Test amaçlı manuel path (normalde main.py'dan gelecek)
    train_model("dataset_klasor_yolu")