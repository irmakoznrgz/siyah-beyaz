import cv2 as cv
import numpy as np
import streamlit as st

# 1. Başlık ekleyelim
st.title("Streamlit Görsel Grileştirme Aracı")

# 2. Resim yükleme bileşeni
uploaded_file = st.file_uploader(
    "Lütfen bir görsel yükleyin...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Yüklenen dosyayı OpenCV formatına (numpy array) dönüştürme
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, 1)

    # Görseli gri tonlamaya çevirme
    gry_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 3. Sonuçları ekranda yan yana göstermek için kolonlar oluşturalım
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Orijinal Görsel")
        # OpenCV BGR okuduğu için Streamlit'e RGB formatında gösteriyoruz
        st.image(cv.cvtColor(image, cv.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("Siyah-Beyaz Görsel")
        st.image(gry_image, use_container_width=True)