from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import uvicorn
import os
import google.generativeai as genai
from dotenv import load_dotenv

# .env yükle
load_dotenv()

app = FastAPI(title="DermaScan AI (Demo Mode)", description="Gemini Vision Destekli Hızlı Analiz")

# API Key Kontrol
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ HATA: GEMINI_API_KEY bulunamadı!")
else:
    genai.configure(api_key=api_key)
    print("✅ Gemini Vision Modu Aktif! (YOLO devredışı)")

@app.get("/")
def home():
    return {"message": "DermaScan Demo Modu Hazır! 🚀"}

@app.post("/analyze")
async def analyze_skin(file: UploadFile = File(...)):
    if not api_key:
        return {"error": "API Key eksik, analiz yapılamıyor."}

    # 1. Resmi Oku
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    print(f"📸 Resim alındı: {file.filename}, Gemini'ye gönderiliyor...")

    # 2. Gemini'ye Gönderilecek Prompt (Hem teşhis hem tavsiye iste)
    prompt = """
    Sen uzman bir dermatologsun. Bu fotoğraftaki kişinin yüzünü analiz et.
    
    GÖREVLER:
    1. Ciltteki problemleri tespit et (Akne, sivilce, kızarıklık, siyah nokta vb. var mı?).
    2. Bunların tahmini sayısını veya yoğunluğunu belirt.
    3. Bu duruma uygun, marka vermeden "içerik odaklı" 3 maddelik kısa bir tavsiye ver.
    4. Çok kısa, profesyonel ama samimi bir dil kullan. Türkçe cevap ver.
    
    Çıktıyı JSON formatına benzer şekilde, başlıklarla ver.
    """

    try:
        # Senin listendeki görsel destekli en iyi model:
        # Eğer hata verirse 'models/gemini-1.5-flash' deneriz.
        model = genai.GenerativeModel('models/gemini-2.5-flash-lite') 
        
        # Modele hem metni hem resmi veriyoruz
        response = model.generate_content([prompt, image])
        
        ai_response = response.text
        print("✅ Analiz Başarılı!")

        return {
            "filename": file.filename,
            "demo_mode": True,
            "ai_analysis": ai_response
        }

    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        return {"error": f"Gemini analizi sırasında hata oluştu: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)