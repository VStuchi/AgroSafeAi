# train.py
from ultralytics import YOLO

# --- Configuração do Treinamento ---

# 1. Carregamos um "cérebro" pré-treinado (yolov8n.pt). 
#    Ele não sabe o que é um EPI, mas já sabe o que são formas e cores.
#    Isso acelera o aprendizado (Transfer Learning).
model = YOLO('yolov8n.pt')

# 2. Caminho para o nosso "livro didático" (o arquivo de configuração do dataset)
#    ATENÇÃO: Verifique se o nome da pasta está correto!
DATASET_PATH = 'deteccao de EPI V3.v4i.yolov8/data.yaml'

# --- Lógica de Treinamento ---
if __name__ == '__main__':
    print("Iniciando treinamento do modelo customizado AgroSafe AI...")
    print(f"Usando o dataset: {DATASET_PATH}")

    try:
        # 3. Iniciamos as "aulas"
        results = model.train(
            data=DATASET_PATH,
            epochs=25,  # Daremos 25 "aulas completas". É o suficiente para um bom MVP.
            imgsz=416,  # Tamanho de imagem otimizado para velocidade.
            device='mps', # Usando a potência da sua GPU M2.
            project='runs/train', # Onde os resultados serão salvos.
            name='AgroSafe_Custom_v1' # Nome do nosso primeiro modelo treinado!
        )

        print("-" * 30)
        print("✅ Treinamento concluído com sucesso!")
        print("Seu primeiro modelo AgroSafe AI foi criado.")
        print("-" * 30)

    except Exception as e:
        print(f"❌ Ocorreu um erro durante o treinamento: {e}")