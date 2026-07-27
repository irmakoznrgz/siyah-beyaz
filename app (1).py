import cv2 as cv
import numpy as np
import streamlit as st
import io

# 1. Sayfa Yapılandırma Ayarları
st.set_page_config(page_title="Gelişmiş Görsel Filtreleme", page_icon="🎨", layout="wide")

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
st.markdown("Yüklediğiniz görsellere anında filtreler uygulayın. **Varsayılan olarak görseliniz siyah-beyaza çevrilir.**")

# 2. Yan Panel (Sidebar) Ayarları - Filtre Seçimi
st.sidebar.title("⚙️ Filtre Ayarları")

filter_type = st.sidebar.selectbox(
    "Uygulamak istediğiniz efekti seçin:",
    (
        "Gri Tonlama (Siyah-Beyaz)", 
        "Orijinal", 
        "Karakalem Çizim",
        "Sepya", 
        "Bulanıklaştırma", 
        "Kenar Bulma (Canny)", 
        "Kabartma (Emboss)",
        "Piksalleştirme",
        "Negatif (Invert)"
    )
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
    if filter_type == "Gri Tonlama (Siyah-Beyaz)":
        processed_image = cv.cvtColor(image_rgb, cv.COLOR_RGB2GRAY)
        
    elif filter_type == "Orijinal":
        processed_image = image_rgb.copy()
        
    elif filter_type == "Karakalem Çizim":
        gray = cv.cvtColor(image_rgb, cv.COLOR_RGB2GRAY)
        inv = cv.bitwise_not(gray)
        blur = cv.GaussianBlur(inv, (21, 21), 0)
        # Renk soldurma tekniği ile karakalem efekti
        processed_image = cv.divide(gray, 255 - blur, scale=256)

    elif filter_type == "Sepya":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv.transform(image_rgb, kernel)
        processed_image = np.clip(sepia, 0, 255).astype(np.uint8)
        
    elif filter_type == "Bulanıklaştırma":
        blur_amount = st.sidebar.slider("Bulanıklık Seviyesi", 1, 99, 15, step=2)
        processed_image = cv.GaussianBlur(image_rgb, (blur_amount, blur_amount), 0)
        
    elif filter_type == "Kenar Bulma (Canny)":
        t1 = st.sidebar.slider("Alt Eşik Değeri", 0, 255, 100)
        t2 = st.sidebar.slider("Üst Eşik Değeri", 0, 255, 200)
        processed_image = cv.Canny(image_rgb, t1, t2)
        
    elif filter_type == "Kabartma (Emboss)":
        kernel_emboss = np.array([[0, -1, -1],
                                  [1,  0, -1],
                                  [1,  1,  0]])
        # Efekti griye çekmek için 128 ekliyoruz
        emboss = cv.filter2D(image_rgb, -1, kernel_emboss)
        processed_image = cv.add(emboss, 128)
        
    elif filter_type == "Piksalleştirme":
        pixel_size = st.sidebar.slider("Piksel Boyutu", 2, 50, 10)
        h, w = image_rgb.shape[:2]
        # Önce küçült, sonra mozaik (yakın komşu) interpolasyonu ile tekrar büyüt
        temp = cv.resize(image_rgb, (w // pixel_size, h // pixel_size), interpolation=cv.INTER_LINEAR)
        processed_image = cv.resize(temp, (w, h), interpolation=cv.INTER_NEAREST)

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
    if len(processed_image.shape) == 3:
        img_to_save = cv.cvtColor(processed_image, cv.COLOR_RGB2BGR)
    else:
        img_to_save = processed_image # Gri, Canny veya Karakalem (tek kanallı)
        
    is_success, buffer = cv.imencode(".png", img_to_save)
    
    if is_success:
        io_buf = io.BytesIO(buffer)
        
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="💾 İşlenmiş Görseli İndir",
                data=io_buf,
                file_name=f"filtrelenmis_{filter_type.replace(' ', '_').lower()}.png",
                mime="image/png",
                use_container_width=True
            )
