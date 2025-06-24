#!/bin/bash
echo "🛠️ Instalando dependências do CLIENTE..."

# Verifica se Python 3 está instalado
if ! command -v python3 >/dev/null; then
    echo "❌ Python3 não encontrado."
    exit 1
else
    echo "✅ Python3 encontrado: $(python3 --version)"
fi

# Verifica pip
if ! command -v pip3 >/dev/null; then
    echo "❌ pip3 não encontrado. Instalando..."
    sudo apt update && sudo apt install -y python3-pip
fi

# Instala dependências
pip3 install --upgrade pip
#pip3 install torch torchvision paho-mqtt --quiet
pip3 install torch --break-system-packages
pip3 install torchvision --break-system-packages
pip3 install paho-mqtt --break-system-packages

# Cria pasta de logs
mkdir -p logs
echo "📁 Pasta 'logs/' criada."

echo "✅ CLIENTE pronto para uso."
