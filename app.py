# app.py - Versão FINAL do MVP (Suporte a Imagem e Vídeo com Playback em Tempo Real)
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import time

# --- Configuração da Página ---
st.set_page_config(
    page_title="AgroSafe AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Título e Descrição ---
st.title("Plataforma de Análise de Segurança - AgroSafe AI 🚀")
st.write("Faça o upload de uma imagem ou vídeo para análise em tempo real de Equipamentos de Proteção Individual (EPIs).")

# --- Barra Lateral (Sidebar) ---
st.sidebar.header("Configurações de Análise")

# --- Carregando o Modelo de IA ---
MODEL_PATH = os.path.join('models', 'best.pt')

@st.cache_resource # Cache do modelo para performance
def load_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar o modelo: {e}")
        return None

model = load_model(MODEL_PATH)
if model is not None:
    st.sidebar.success("Modelo carregado com sucesso!")
else:
    st.sidebar.error("Verifique o caminho do modelo e reinicie a aplicação.")
    st.stop()

# --- Controles da Barra Lateral ---
confidence_threshold = st.sidebar.slider(
    "Limiar de Confiança", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
iou_threshold = st.sidebar.slider(
    "Limiar de Sobreposição (IOU)", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
st.sidebar.markdown("---")

# --- Interface de Upload ---
uploaded_file = st.sidebar.file_uploader(
    "Escolha uma imagem ou vídeo...", 
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi"]
)

# --- Lógica de Processamento e Exibição ---
if uploaded_file is not None:
    file_type = uploaded_file.type
    
    # Processamento para IMAGEM
    if file_type.startswith('image/'):
        image = Image.open(uploaded_file)
        results = model(image, device='mps', conf=confidence_threshold, iou=iou_threshold)
        annotated_image = results[0].plot()
        annotated_image_rgb = annotated_image[..., ::-1]
        st.image(annotated_image_rgb, caption="Imagem Analisada", use_container_width=True)

    # Processamento para VÍDEO (NOVA LÓGICA EM TEMPO REAL)
    elif file_type.startswith('video/'):
        st.subheader("Análise de Vídeo em Tempo Real")
        
        # Salva o vídeo temporariamente para o OpenCV poder abri-lo
        temp_video_path = os.path.join(".", uploaded_file.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Abre o vídeo
        cap = cv2.VideoCapture(temp_video_path)

        # Cria um placeholder na tela para exibir o vídeo
        image_placeholder = st.empty()
        
        # Botão para parar a análise
        stop_button_pressed = st.button("Parar Análise")

        # Calcula o tempo de espera entre os frames para simular a velocidade original do vídeo
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            sleep_time = 1 / fps if fps > 0 else 0
        except:
            sleep_time = 0.03 # Valor padrão caso não consiga ler o FPS

        while cap.isOpened() and not stop_button_pressed:
            ret, frame = cap.read()
            if not ret:
                st.write("O vídeo terminou.")
                break
            
            # Processa o frame com o modelo
            results = model(frame, device='mps', conf=confidence_threshold, iou=iou_threshold)
            annotated_frame = results[0].plot()
            
            # Converte a cor do frame para RGB para exibição no Streamlit
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Exibe o frame processado no placeholder
            image_placeholder.image(annotated_frame_rgb, caption="Vídeo em Análise", use_container_width=True)
            
            # Espera um pouco para simular o FPS original
            time.sleep(sleep_time)

        # Libera os recursos
        cap.release()
        os.remove(temp_video_path)