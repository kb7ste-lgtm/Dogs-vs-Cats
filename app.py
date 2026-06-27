import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Configuration de la page
st.set_page_config(
    page_title="IA Vision | Chats et Chiens",
    page_icon="🐾",
    layout="centered"
)

# 2. CSS Complet et Fixé pour la lisibilité des textes
st.markdown("""
    <style>
    /* Image de fond globale avec filtre sombre */
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.85)), 
                    url("https://images.unsplash.com/photo-1543466835-00a7907e9de1?q=80&w=1974&auto=format&fit=crop") !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
    }
    
    /* Disparition de la bande blanche supérieure */
    header, .stHeader, [data-testid="stHeader"] {
        background: transparent !important;
        background-color: transparent !important;
        display: none !important;
    }
    .block-container {
        padding-top: 2rem !important;
    }
    
    /* On force les titres et textes principaux en blanc */
    h1, h2, h3, p, label, .stMarkdown {
        color: #FFFFFF !important;
    }
    
    /* Style de la zone de dépôt d'image */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.07) !important;
        border: 2px dashed #38BDF8 !important; 
        border-radius: 12px !important;
        padding: 10px !important;
    }
    
    /* Bouton Upload bleu dynamique */
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #38BDF8 !important; 
        color: #0F172A !important; 
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.4) !important;
    }
    
    /* CORRECTION : Force le petit texte d'info de l'uploader en gris foncé pour être visible */
    [data-testid="stFileUploaderDropzone"] small {
        color: #475569 !important;
        font-weight: 500 !important;
    }
    
    /* Style de la boîte de résultat */
    .prediction-box {
        padding: 22px;
        border-radius: 16px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Titres de l'application
st.markdown("<h1 style='text-align: center;'>🐾 IA Vision : Chats vs Chiens</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8 !important; font-size: 16px;'>Détection automatisée par réseau de neurones</p>", unsafe_allow_html=True)
st.markdown("---")

# 4. Chargement du modèle Keras
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('model_cats_dogs_tl.keras')

with st.spinner("Connexion aux synapses de l'IA..."):
    model = load_my_model()

st.markdown("<br>", unsafe_allow_html=True)

# 5. Zone d'upload d'image
uploaded_file = st.file_uploader("Glissez ou sélectionnez une photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.markdown("---")
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image analysée", use_container_width=True)
    
    with col2:
        with st.status("Calcul des probabilités...", expanded=False) as status:
            img_resized = image.resize((150, 150))
            img_array = np.array(img_resized)
            
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
                
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array)[0][0]
            status.update(label="Analyse terminée !", state="complete")
        
        st.markdown("### Verdict de l'IA")
        
        if prediction > 0.5:
            confiance = prediction
            st.markdown(
                f"<div class='prediction-box' style='background-color: #064E3B; color: #A7F3D0; border: 2px solid #047857;'> "
                f"🐾 C'est un CHIEN !!<br><span style='font-size: 15px; font-weight: normal; color: #34D399;'>Certitude : {confiance:.2%}</span>"
                f"</div>", 
                unsafe_allow_html=True
            )
            st.balloons()
        else:
            confiance = 1 - prediction
            st.markdown(
                f"<div class='prediction-box' style='background-color: #0C4A6E; color: #BAE6FD; border: 2px solid #0369A1;'> "
                f"🐾 C'est un CHAT !!<br><span style='font-size: 15px; font-weight: normal; color: #38BDF8;'>Certitude : {confiance:.2%}</span>"
                f"</div>", 
                unsafe_allow_html=True
            )
            st.balloons()
else:
    st.info("Astuce : Utilisez une photo bien éclairée pour une précision maximale.")