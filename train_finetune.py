# train_finetune.py
from ultralytics import YOLO

# 1. Carregamos o NOSSO MELHOR MODELO ATUAL como ponto de partida
# Ele já sabe tudo sobre capacetes, pessoas, etc.
model = YOLO('models/best.pt') 

# 2. Apontamos para o nosso NOVO e PEQUENO dataset de reforço
DATASET_PATH = 'AgroSafe-Reforco.v2-roboflow-instant-2--eval-.yolov8/data.yaml'

# --- Inicia o Treinamento de Ajuste Fino ---
if __name__ == '__main__':
    print("Iniciando AJUSTE FINO do modelo AgroSafe AI...")

    # 3. Treinamos por POUCAS épocas. Só queremos que ele aprenda
    # as novas classes sem esquecer as antigas.
    model.train(
        data=DATASET_PATH,
        epochs=30,  # Geralmente menos épocas são necessárias para fine-tuning
        imgsz=640,  # Podemos usar uma resolução maior, pois o dataset é pequeno
        name='AgroSafe_Custom_v2_Especialista'
    )