#!/bin/bash

echo "📦 Criando ambiente virtual em ./venv ..."
python3 -m venv venv

echo "⚙️ Ativando ambiente virtual..."
source venv/bin/activate

echo "📥 Instalando dependências do requirements.txt..."
pip install -r LLM/requirements.txt

echo "✅ Ambiente pronto!"