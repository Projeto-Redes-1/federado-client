#!/bin/bash

echo "🛠️ Iniciando instalação para CLIENTE..."

# Verifica Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 não encontrado. Por favor, instale antes de continuar."
    exit 1
else
    echo "✅ Python3 OK."
fi

# Verifica pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 não encontrado. Instalando..."
    sudo apt update
    sudo apt install -y python3-pip
fi

# Instala dependências do cliente
echo "📦 Instalando pacotes Python para cliente..."
pip3 install --upgrade pip
pip3 install torch torchvision paho-mqtt --quiet

# Cria pasta de logs
mkdir -p src/logs
echo "📁 Pasta de logs criada em src/logs"

echo "✅ Instalação para CLIENTE concluída!"
