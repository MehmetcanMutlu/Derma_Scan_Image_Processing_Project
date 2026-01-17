import google.generativeai as genai
import os
from dotenv import load_dotenv

# Şifreleri yükle
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ API Key bulunamadı! .env dosyasını kontrol et.")
else:
    genai.configure(api_key=api_key)
    print("\n🔍 --- SENİN ANAHTARININ GÖRDÜĞÜ MODELLER ---")
    try:
        bulundu = False
        for m in genai.list_models():
            # Sadece metin üretebilen modelleri göster
            if 'generateContent' in m.supported_generation_methods:
                print(f"✅ {m.name}")
                bulundu = True
        
        if not found:
            print("⚠️ Hiçbir model listelenemedi. Erişim yetkisi sorunu olabilir.")
            
    except Exception as e:
        print(f"❌ HATA: {e}")
    print("---------------------------------------------")