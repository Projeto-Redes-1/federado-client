import os
import time
import pickle
from datetime import datetime
import paho.mqtt.client as mqtt
from client import Client
from logger_utils import setup_logger, get_host_info

# Define o ID do cliente com base na variável de ambiente, padrão 0 se não existir
CLIENT_ID = int(os.getenv("CLIENT_ID", 0))
# Define o IP do broker MQTT (servidor), padrão localhost
BROKER_IP = os.getenv("BROKER_IP", "localhost")

# Tópico MQTT onde o cliente vai publicar seus parâmetros locais após treino
PARAMS_TOPIC = f"fed/client/{CLIENT_ID}/params"
# Tópico MQTT onde o cliente espera receber o modelo global para treino local
GLOBAL_TOPIC = "fed/global/params"

# Configura o logger com nome e arquivo específico para esse cliente
logger = setup_logger(f"client_{CLIENT_ID}", f"logs/client_{CLIENT_ID}.log")

# Pega hostname e IP da máquina para logs e prints
hostname, ip = get_host_info()

# Log inicial informando que o cliente iniciou e printa no console
logger.info(f"1. Cliente {CLIENT_ID} iniciado em {hostname} ({ip}) às {datetime.now()}")
print(f"[INFO] 1. Cliente {CLIENT_ID} iniciado em {hostname} ({ip}) às {datetime.now()}")

# Cria cliente MQTT explicitando client_id e protocolo para evitar warning
mqtt_client = mqtt.Client(client_id=str(CLIENT_ID), protocol=mqtt.MQTTv311)

# Callback chamado quando conecta no broker MQTT
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"[INFO] Conectado com sucesso ao broker MQTT.")
        logger.info("Conectado com sucesso ao broker MQTT.")
        # Subscrição feita somente após conexão
        client.subscribe(GLOBAL_TOPIC)
        print(f"[INFO] Subscrito no tópico: {GLOBAL_TOPIC}")
        logger.info(f"Subscrito no tópico: {GLOBAL_TOPIC}")
    else:
        print(f"[ERROR] Falha na conexão, código {rc}")
        logger.error(f"Falha na conexão, código {rc}")

# Callback chamado quando a subscrição é confirmada (opcional para debug)
def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    print(f"[INFO] Inscrição confirmada.")
    logger.info("Inscrição confirmada.")

# Callback chamado quando recebe mensagem de algum tópico subscrito
def on_message(client, userdata, msg):
    # Log e print informando recebimento do modelo global
    logger.info(f"2. [{datetime.now()}] Parâmetros globais recebidos ({len(msg.payload)} bytes).")
    print(f"[INFO] 2. [{datetime.now()}] Parâmetros globais recebidos ({len(msg.payload)} bytes).")

    # Salva o conteúdo recebido em arquivo local 'global_parameters.pkl'
    with open("global_parameters.pkl", "wb") as f:
        f.write(msg.payload)
    logger.info("3. Parâmetros globais salvos em 'global_parameters.pkl'.")
    print("[INFO] 3. Parâmetros globais salvos em 'global_parameters.pkl'.")

    # Desconecta o cliente MQTT pois já recebeu o que precisava
    client.disconnect()
    print("[INFO] Cliente MQTT desconectado após receber parâmetros globais.")

# Callback para quando desconecta (opcional para debug)
def on_disconnect(client, userdata, rc):
    print(f"[INFO] Cliente MQTT desconectado com código {rc}.")
    logger.info(f"Cliente MQTT desconectado com código {rc}.")

# Associa os callbacks ao cliente MQTT
mqtt_client.on_connect = on_connect
mqtt_client.on_subscribe = on_subscribe
mqtt_client.on_message = on_message
mqtt_client.on_disconnect = on_disconnect

# Conecta ao broker MQTT no IP e porta padrão 1883
mqtt_client.connect(BROKER_IP, 1883, 60)
print(f"[INFO] Conectando ao broker MQTT em {BROKER_IP}:1883")

# Inicia o loop do MQTT para ficar escutando mensagens assincronamente
mqtt_client.loop_start()
print("[INFO] Loop MQTT iniciado para aguardar modelo global.")

# Marca o tempo de início da espera pelo modelo global
start_wait = time.time()

# Fica em loop esperando até o arquivo 'global_parameters.pkl' ser criado,
# ou seja, até receber o modelo global e salvá-lo em disco
while not os.path.exists("global_parameters.pkl"):
    logger.info("4. Aguardando parâmetros globais...")
    print("[INFO] 4. Aguardando parâmetros globais...")
    time.sleep(2)  # Espera 2 segundos antes de checar novamente

# Calcula quanto tempo levou para receber o modelo global
wait_time = time.time() - start_wait
logger.info(f"5. Modelo global recebido após {wait_time:.2f}s de espera.")
print(f"[INFO] 5. Modelo global recebido após {wait_time:.2f}s de espera.")

# Abre o arquivo salvo com os parâmetros globais para carregar os dados
with open("global_parameters.pkl", "rb") as f:
    parameters = pickle.load(f)

# Marca o tempo do início do treino local
start_train = time.time()
logger.info("6. Iniciando treinamento local...")
print("[INFO] 6. Iniciando treinamento local...")

# Cria o objeto Client local com o ID do cliente
client = Client(CLIENT_ID)

# Chama o método de treino local passando os parâmetros globais recebidos
updated_parameters = client.train(parameters)

# Calcula a duração do treino local
duration = time.time() - start_train
logger.info(f"7. Treinamento local concluído em {duration:.2f} segundos às {datetime.now()}.")
print(f"[INFO] 7. Treinamento local concluído em {duration:.2f} segundos às {datetime.now()}.")

# Serializa os parâmetros atualizados para enviar via MQTT
payload = pickle.dumps(updated_parameters)

# Publica os parâmetros treinados no tópico específico para esse cliente
mqtt_client.publish(PARAMS_TOPIC, payload)
logger.info(f"8. Enviou {len(payload)} bytes para o servidor no tópico: {PARAMS_TOPIC}.")
print(f"[INFO] 8. Enviou {len(payload)} bytes para o servidor no tópico: {PARAMS_TOPIC}.")

# Salva localmente o arquivo dos parâmetros treinados para inspeção futura
with open(f"client_{CLIENT_ID}_parameters.pkl", "wb") as f:
    f.write(payload)
logger.info(f"9. Parâmetros locais salvos como 'client_{CLIENT_ID}_parameters.pkl'.")
print(f"[INFO] 9. Parâmetros locais salvos como 'client_{CLIENT_ID}_parameters.pkl'.")

# Para o loop MQTT pois terminou o processo
mqtt_client.loop_stop()
print("[INFO] Loop MQTT parado.")

logger.info("10. Cliente finalizado.")
print("[INFO] 10. Cliente finalizado.")
