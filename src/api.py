from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn
import os
from doctor import get_skin_advice 

app = FastAPI(title="DermaScan Microservice API")

# İndirdiğin modelin yolu (Yolu kontrol et!)
MODEL_PATH = "runs/detect/akne_modeli/weights/best.pt"

if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print(f"✅ Microservice Hazır: Model ({MODEL_PATH}) yüklendi.")
else:
    print(f"❌ HATA: Model bulunamadı: {MODEL_PATH}")
    model = None

@app.post("/analyze")
async def analyze_skin(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model yüklenemedi, analiz yapılamıyor."}

    # 1. Resmi Oku
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # 2. YOLO Analizi (Hassasiyet: %15)
    results = model.predict(image, conf=0.15)
    
    # 3. Koordinatları ve Türleri Çıkar
    detections = []
    for box in results[0].boxes:
        cord = box.xyxy[0].tolist() # [x1, y1, x2, y2] formatında koordinat
        conf = round(float(box.conf[0]), 2)
        cls_name = model.names[int(box.cls[0])]
        
        detections.append({
            "type": cls_name,
            "confidence": conf,
            "bbox": cord # Kutu çizmek için bu lazım
        })

    # 4. Doktora Danış
    advice = get_skin_advice(len(detections), detections)

    return {
        "detection_count": len(detections),
        "results": detections,
        "doctor_advice": advice
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)