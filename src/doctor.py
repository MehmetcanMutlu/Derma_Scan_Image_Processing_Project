import os
import google.generativeai as genai
from dotenv import load_dotenv

# .env yükle
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    print("⚠️ UYARI: API Key bulunamadı!")

def get_skin_advice(detection_count, detection_list):
    """
    YOLO sonuçlarını alır, Gemini 2.5'e sorar ve tavsiye döner.
    """
    if not api_key:
        return "Yapay zeka anahtarı eksik."
    
    if detection_count == 0:
        return "Cildiniz harika görünüyor! ✨ Mevcut rutininizi koruyun ve bol su için. 💧"

    # SENİN LİSTENDEN SEÇTİĞİMİZ MODEL 🚀
    model_name = 'models/gemini-2.5-flash-lite'

    prompt = f"""
    Sen uzman bir dermatologsun. Aşağıdaki cilt analizi sonuçlarına göre kullanıcıya tavsiye ver.
    
    ANALİZ SONUÇLARI:
    - Tespit Sayısı: {detection_count}
    - Detaylar: {detection_list}

    GÖREV:
    1. Durumu samimi bir dille yorumla.
    2. Tespit edilen sorunlara özel (akne, siyah nokta vb.) 3 maddelik içerik tavsiyesi ver (Marka verme, etken madde söyle örn: Salisilik asit).
    3. Motive edici kısa bir kapanış yap.
    
    Cevap Türkçe ve emojili olsun.
    """

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Tavsiye oluşturulamadı. Hata: {str(e)}"

# --- TEST ---
if __name__ == "__main__":
    print(f"⏳ Gemini ({'models/gemini-2.5-flash-lite'}) düşünüyor...")
    test_data = [{"type": "acne", "confidence": 0.95}, {"type": "blackhead", "confidence": 0.88}]
    
    cevap = get_skin_advice(2, test_data)
    print("\n⬇️ DOKTORUN TAVSİYESİ ⬇️")
    print(cevap)