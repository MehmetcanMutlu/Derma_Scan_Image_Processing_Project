from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn
import os

app = FastAPI(title="DermaScan AI API", description="Cilt Analiz ve Sivilce Tespit API'si")

# Eğitilen modelin tam yolu
# Klasör adının 'akne_modeli' olduğundan emin ol.
MODEL_PATH = "runs/detect/akne_modeli/weights/best.pt"

# Model dosyasının varlığını kontrol edelim
if not os.path.exists(MODEL_PATH):
    print(f"UYARI: Model dosyası bulunamadı: {MODEL_PATH}")
    print("Lütfen 'runs/detect/' klasörünü kontrol edip doğru yolu yaz.")
    model = None
else:
    model = YOLO(MODEL_PATH)
    print("✅ Model başarıyla yüklendi!")

@app.get("/")
def home():
    return {"message": "DermaScan AI Çalışıyor! /analyze endpointini kullanın."}

@app.post("/analyze")
async def analyze_skin(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model yüklenemedi. Dosya yolunu kontrol edin."}

    # 1. Gelen resmi oku
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # 2. Modele gönder (Tahmin)
    # GÜNCELLEME: conf=0.10 yaptık (Hassasiyeti artırdık)
    results = model.predict(image, conf=0.10)
    
    # 3. Sonuçları işle
    result_list = []
    # results bir liste döner, ilk eleman bizim resmimiz
    for box in results[0].boxes:
        # Sınıf ID'sini (0, 1, 2...) al
        cls_id = int(box.cls[0])
        # Sınıf ismini modelin isim listesinden çek (akne, siyah nokta vb.)
        cls_name = model.names[cls_id]
        
        result_list.append({
            "type": cls_name,
            "confidence": round(float(box.conf[0]), 2), # % kaç emin?
            "coordinates": box.xywh[0].tolist() # Koordinatlar
        })

    # 4. JSON Cevabı Dön
    return {
        "filename": file.filename,
        "count": len(result_list),
        "detections": result_list,
        # Burası ileride Gemini/GPT ile dolacak, şimdilik statik
        "advice": "Analiz tamamlandı. Detaylı cilt bakım tavsiyesi için LLM modülü bekleniyor."
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)