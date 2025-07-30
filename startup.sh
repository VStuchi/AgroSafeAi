#!/bin/bash

echo "--- [AgroSafe AI] Iniciando script de inicialização customizado ---"

echo "--- [AgroSafe AI] Instalando dependências do requirements.txt ---"
pip install -r requirements.txt

echo "--- [AgroSafe AI] Dependências instaladas com sucesso. ---"

echo "--- [AgroSafe AI] Iniciando a aplicação Streamlit na porta 8000 ---"
streamlit run app.py --server.port 8000 --server.enableCORS false --server.address=0.0.0.0