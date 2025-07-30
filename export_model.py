# export_model.py - VERSÃO CORRIGIDA FINAL
from roboflow import Roboflow

# --- Configuração ---
# Use as informações do seu projeto no Roboflow
API_KEY = "HhcygV5EN7mWmH9M0T30" 
WORKSPACE_ID = "stuchi" # Use o ID do seu workspace
PROJECT_ID = "meu-projeto-epi"  # Use o ID do seu projeto
VERSION_NUMBER = 1          # O número da versão que você quer

# --- Lógica de Download ---
try:
    print("Iniciando conexão com o Roboflow...")
    rf = Roboflow(api_key=API_KEY)
    
    print(f"Acessando o seu workspace '{WORKSPACE_ID}'...")
    project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
    
    print(f"Solicitando a versão {VERSION_NUMBER} no formato 'yolov8'...")
    version = project.version(VERSION_NUMBER)
    
    # A função correta é .download(), que baixa o dataset no formato especificado.
    # O modelo treinado (best.pt) virá junto com os dados neste formato.
    dataset = version.download("yolov8")

    print("-" * 30)
    print("✅ Download concluído com sucesso!")
    # A variável 'dataset.location' nos dirá exatamente onde os arquivos foram salvos
    print(f"Os arquivos foram salvos na pasta: '{dataset.location}'")
    print("Verifique este caminho no seu projeto. Dentro dele, você encontrará a pasta 'weights' com o arquivo 'best.pt'.")
    print("-" * 30)

except Exception as e:
    print(f"❌ Ocorreu um erro: {e}")