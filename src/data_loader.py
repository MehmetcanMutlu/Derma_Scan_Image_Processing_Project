import os
from roboflow import Roboflow
from dotenv import load_dotenv

# .env dosyasındaki şifreleri yükle
load_dotenv()

def download_dataset():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("API Key bulunamadı! .env dosyasını kontrol et.")

    print("🚀 Veri seti indiriliyor...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("fyp-acne-detection").project("acne-detection-yolov8")
    version = project.version(2)
    dataset = version.download("yolov8")
    
    # İndirilen klasörün yolunu döndür (train.py'da kullanmak için)
    return dataset.location

if __name__ == "__main__":
    download_dataset()