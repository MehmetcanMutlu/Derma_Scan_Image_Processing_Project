import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io

# API Adresi (Back-end)
API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="DermaScan Client", layout="wide")

st.title("🔬 DermaScan - Canlı Model Testi")
st.info("Bu arayüz, arka planda çalışan YOLO servisine bağlanır ve gerçek sonucu gösterir.")

uploaded_file = st.file_uploader("Test edilecek fotoğrafı yükle...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    # Resmi Göster (Orjinal)
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Orjinal Fotoğraf", use_container_width=True)

    # Butona basınca API'ye git
    if st.button("🚀 API'ye Gönder ve Analiz Et"):
        with st.spinner("Microservice ile haberleşiliyor..."):
            try:
                # 1. Resmi Byte'a çevirip gönder
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                img_byte_arr = img_byte_arr.getvalue()

                files = {"file": ("image.jpg", img_byte_arr, "image/jpeg")}
                response = requests.post(API_URL, files=files)
                
                # 2. Cevabı Al
                if response.status_code == 200:
                    data = response.json()
                    detections = data["results"]
                    
                    # 3. Resmin Üzerine Kutucukları Çiz 🎨
                    draw_image = image.copy()
                    draw = ImageDraw.Draw(draw_image)
                    
                    for det in detections:
                        box = det["bbox"] # [x1, y1, x2, y2]
                        label = f"{det['type']} ({det['confidence']})"
                        
                        # Kutu Çiz (Kırmızı)
                        draw.rectangle(box, outline="red", width=3)
                        # Yazı Yaz (İsteğe bağlı, basit olsun diye sadece kutu çiziyoruz şimdilik)
                    
                    # 4. Sonucu Sağ Tarafa Bas
                    with col2:
                        st.image(draw_image, caption=f"YOLO Tespiti ({data['detection_count']} Sivilce)", use_container_width=True)
                        
                    st.success("Analiz Tamamlandı!")
                    st.markdown("### 👨‍⚕️ Doktorun Yorumu:")
                    st.write(data["doctor_advice"])
                    
                else:
                    st.error(f"API Hatası: {response.status_code}")

            except Exception as e:
                st.error(f"Bağlantı Hatası: {e}. API (api.py) çalışıyor mu?")