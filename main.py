from src.data_loader import download_dataset
from src.train import train_model

def main():
    print("--- DermaScan AI Başlatılıyor ---")
    
    # 1. Veriyi İndir
    try:
        dataset_path = download_dataset()
        print(f"📂 Veri seti şuraya indi: {dataset_path}")
    except Exception as e:
        print(f"Hata: {e}")
        return

    # 2. Modeli Eğit
    # İstersen kullanıcıya sorabilirsin
    cevap = input("Modeli şimdi eğitmek ister misin? (e/h): ")
    if cevap.lower() == 'e':
        train_model(dataset_path, epochs=10) # Deneme için 10 epoch yeter
    else:
        print("Eğitim atlandı.")

if __name__ == "__main__":
    main()