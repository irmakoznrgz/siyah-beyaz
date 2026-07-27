import cv2 as cv
import numpy as np
import streamlit as st
import io

# 1. Sayfa Yapılandırması ve Şık Görünüm Ayarları
st.set_page_config(page_title="Gelişmiş Görsel Filtreleme", page_icon="🎨", layout="wide")

# Butonları ve arayüzü daha şık yapmak için özel CSS
st.markdown("""
    <style>
    .stDownloadButton>button {
        border-radius: 20px;
        background-color: #FF4B4B;
        color: white;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .stDownloadButton>button:hover {
        background-color: #FF6B6B;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎨 Streamlit Gelişmiş Görsel Filtreleme Aracı")
st.markdown("Yüklediğiniz görsellere çeşitli filtreler uygulayın ve sonuçları anında görün.")

# 2. Yan Panel (Sidebar) Ayarları - Filtre Seçimi
st.sidebar.title("⚙️ Filtre Ayarları")
filter_type = st.sidebar.selectbox(
    "Uygulamak istediğiniz efekti seçin:",
    ("Orijinal", "Gri Tonlama", "Sepya", "Bulanıklaştırma", "Kenar Bulma (Canny)", "Negatif (Invert)")
)

# 3. Resim yükleme bileşeni
uploaded_file = st.file_uploader(
    "Lütfen bir görsel yükleyin... (jpg, jpeg, png)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Yüklenen dosyayı OpenCV formatına dönüştürme
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv.imdecode(file_bytes, 1)
    
    # İşlemleri kolaylaştırmak için BGR'den RGB'ye çeviriyoruz
    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    processed_image = image_rgb.copy()

    # 4. Seçilen Filtrenin Uygulanması
    if filter_type == "Gri Tonlama":
        processed_image = cv.cvtColor(image_rgb, cv.COLOR_RGB2GRAY)
        
    elif filter_type == "Sepya":
        # Sepya efekti için matris dönüşümü
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv.transform(image_rgb, kernel)
        processed_image = np.clip(sepia, 0, 255).astype(np.uint8)
        
    elif filter_type == "Bulanıklaştırma":
        # Slider üzerinden bulanıklık şiddetini ayarlama (tek sayı olmalı)
        blur_amount = st.sidebar.slider("Bulanıklık Seviyesi", 1, 99, 15, step=2)
        processed_image = cv.GaussianBlur(image_rgb, (blur_amount, blur_amount), 0)
        
    elif filter_type == "Kenar Bulma (Canny)":
        # Slider üzerinden kenar bulma hassasiyetini ayarlama
        t1 = st.sidebar.slider("Alt Eşik Değeri", 0, 255, 100)
        t2 = st.sidebar.slider("Üst Eşik Değeri", 0, 255, 200)
        processed_image = cv.Canny(image_rgb, t1, t2)
        
    elif filter_type == "Negatif (Invert)":
        processed_image = cv.bitwise_not(image_rgb)

    # 5. Sonuçları ekranda yan yana gösterme
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Orijinal Görsel")
        st.image(image_rgb, use_container_width=True)

    with col2:
        st.subheader(f"✨ İşlenmiş Görsel ({filter_type})")
        st.image(processed_image, use_container_width=True)

    # 6. İşlenmiş Görseli İndirme Butonu
    # OpenCV kaydetmeden önce formatı tekrar ayarlamalıyız
    if len(processed_image.shape) == 3:
        img_to_save = cv.cvtColor(processed_image, cv.COLOR_RGB2BGR)
    else:
        img_to_save = processed_image # Gri veya Canny (tek kanallı)
        
    is_success, buffer = cv.imencode(".png", img_to_save)
    
    if is_success:
        io_buf = io.BytesIO(buffer)
        
        # Butonu ortalamak için boş kolonlar kullanıyoruz
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True) # Üstten biraz boşluk
            st.download_button(
                label="💾 İşlenmiş Görseli İndir",
                data=io_buf,
                file_name=f"filtrelenmis_{filter_type.replace(' ', '_').lower()}.png",
                mime="image/png",
                use_container_width=True
            )
