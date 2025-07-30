# app_live.py - Versão 3.0 do MVP AgroSafe AI (Análise em Tempo Real)
import streamlit as st
from ultralytics import YOLO
import os
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoFrame
import av

# --- Configuração da Página ---
st.set_page_config(
    page_title="AgroSafe AI - Análise Ao Vivo",
    page_icon="🎥",
    layout="wide"
)

# --- Título ---
st.title("Monitoramento em Tempo Real - AgroSafe AI 👁️")
st.write("Clique em 'START' para iniciar a análise da sua webcam.")

# --- Carregando o Modelo de IA ---
MODEL_PATH = os.path.join('models', 'custom_v1', 'best.pt')

@st.cache_resource # Cache do modelo para performance
def load_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

model = load_model(MODEL_PATH)

# --- Controles da Barra Lateral ---
st.sidebar.header("Configurações da Análise")
confidence_threshold = st.sidebar.slider(
    "Limiar de Confiança", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
iou_threshold = st.sidebar.slider(
    "Limiar de Sobreposição (IOU)", min_value=0.0, max_value=1.0, value=0.45, step=0.05)

# --- Lógica de Processamento em Tempo Real ---

if model is None:
    st.error("Modelo não carregado. A aplicação não pode continuar.")
else:
    # A "mágica" acontece nesta função de callback
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        # 1. Converte o frame do formato da biblioteca para o formato do OpenCV
        img = frame.to_ndarray(format="bgr24")

        # 2. Roda a detecção no frame
        results = model(img, device='mps', conf=confidence_threshold, iou=iou_threshold)
        
        # 3. Desenha as caixas e os rótulos no frame
        annotated_frame = results[0].plot()

        # 4. Retorna o frame processado para ser exibido na tela
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    # Componente do Streamlit que liga a webcam e chama nossa função para cada frame
    webrtc_streamer(
        key="live_analysis",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.markdown("""
        **Como usar:**
        1. Permita o acesso à sua câmera quando o navegador solicitar.
        2. Clique em **START** para iniciar o vídeo.
        3. A IA analisará o vídeo em tempo real.
        4. Clique em **STOP** para encerrar.
    """)