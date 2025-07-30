# evaluate_models.py
from ultralytics import YOLO

# --- CONFIGURAÇÃO ---
# 1. Caminho para o modelo antigo (pré-treinado)
OLD_MODEL_PATH = 'models/best.pt' 

# 2. Caminho para o seu NOVO modelo (customizado)
# ATENÇÃO: Verifique se este caminho está correto
NEW_MODEL_PATH = 'models/custom_v1/best.pt'

# 3. Caminho para o data.yaml do SEU dataset que foi usado no treino
# Este dataset tem as imagens de validação que serão usadas como "prova"
DATA_YAML_PATH = 'deteccao de EPI V3.v1i.yolov8/data.yaml' # Verifique o nome da pasta

# --- AVALIAÇÃO ---
print("--- AVALIANDO MODELO ANTIGO (PRÉ-TREINADO) ---")
try:
    old_model = YOLO(OLD_MODEL_PATH)
    old_model.val(data=DATA_YAML_PATH, name='old_model_eval')
except Exception as e:
    print(f"Erro ao avaliar modelo antigo: {e}")


print("\n" + "="*50 + "\n")


print("--- AVALIANDO NOVO MODELO (CUSTOMIZADO) ---")
try:
    new_model = YOLO(NEW_MODEL_PATH)
    new_model.val(data=DATA_YAML_PATH, name='new_model_eval')
except Exception as e:
    print(f"Erro ao avaliar novo modelo: {e}")