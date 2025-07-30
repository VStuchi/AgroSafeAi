# check_model.py
from ultralytics import YOLO
import os

# --- CONFIGURAÇÃO ---
# Coloque o caminho exato para o arquivo .pt que você está usando no seu app.py
MODEL_PATH = os.path.join('models', 'best.pt') 

# --- LÓGICA DE INSPEÇÃO ---
try:
    print(f"🔎 Carregando o modelo de: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Pega o dicionário de classes (ex: {0: 'Hardhat', 1: 'Mask', ...})
    class_names_dict = model.names
    
    # Garante que os nomes estejam na ordem correta dos IDs (0, 1, 2...)
    # e pega apenas os nomes
    class_list = [class_names_dict[i] for i in sorted(class_names_dict.keys())]
    
    # Junta todos os nomes da lista em uma única string, separados por vírgula
    comma_separated_names = ",".join(class_list)
    
    # Imprime o resultado final, pronto para copiar
    print(comma_separated_names)

except Exception as e:
    print(f"❌ Ocorreu um erro ao carregar o modelo: {e}")