#!/bin/bash

echo "🚀 Iniciando cliente federado..."

# Define variáveis de ambiente
export CLIENT_ID=${1:-0}  # se não passar argumento, usa 0
export BROKER_IP=${2:-"192.168.1.100"}  # IP do servidor MQTT

echo "ℹ️ CLIENT_ID: $CLIENT_ID"
echo "ℹ️ BROKER_IP: $BROKER_IP"

# Ativa ambiente virtual se quiser (opcional)

# Executa o cliente MQTT
python3 src/client_mqtt.py
