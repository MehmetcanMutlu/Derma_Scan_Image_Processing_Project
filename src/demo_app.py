import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image

# 1. Ayarları Yükle
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Sayfa Ayarları (Sekme adı ve ikon)
st.set_page_config(page_title="DermaScan AI", page_icon="🩺", layout="wide")

# Başlık
st.title("🩺 DermaScan AI - Akıllı Cilt Analizi")
st.markdown("Cilt fotoğrafınızı yükleyin, yapay zeka saniyeler içinde analiz etsin.")

# Yan Menü (Sidebar)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=100)
    st.header("Sistem Durumu")
    if api_key:
        st.success("✅ AI Motoru Hazır")
        genai.configure(api_key=api_key)
    else:
        st.error("❌ API Anahtarı Bulunamadı!")
        st.stop()

# 2. Dosya Yükleme Alanı
uploaded_file = st.file_uploader("Analiz için bir fotoğraf yükleyin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Ekranı ikiye böl (Sol: Resim, Sağ: Analiz)
    col1, col2 = st.columns([1, 1])

    # SOL KOLON: Resim
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Yüklenen Fotoğraf', use_container_width=True)
        
        # Analiz Butonu
        analyze_button = st.button("🔍 Cildi Analiz Et", type="primary")

    # SAĞ KOLON: Sonuçlar
    with col2:
        if analyze_button:
            with st.spinner("Yapay zeka cildi inceliyor, lütfen bekleyin..."):
                try:
                    # Model Seçimi (Senin çalışan modelin)
                    model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
                    
                    prompt = """
                    Sen uzman bir dermatologsun. Bu fotoğrafı analiz et.
                    Cevabı doğrudan, süslü bir Markdown formatında ver (JSON olmasın).
                    
                    Şu başlıkları kullan:
                    ### 🧐 Tespit Edilen Durum
                    (Buraya ne gördüğünü kısaca yaz)
                    
                    ### 📊 Yoğunluk
                    (Hafif/Orta/Yüksek değerlendirmesi yap)
                    
                    ### 💡 Uzman Tavsiyeleri
                    (3 madde halinde içerik önerileri ver)
                    
                    ### 🩺 Doktorun Notu
                    (Kısa, motive edici bir kapanış)
                    """
                    
                    response = model.generate_content([prompt, image])
                    
                    # Sonucu Ekrana Bas
                    st.markdown(response.text)
                    st.success("Analiz tamamlandı! ✨")
                    
                except Exception as e:
                    st.error(f"Bir hata oluştu: {str(e)}")
        else:
            st.info("Analizi başlatmak için butona tıklayın 👈")