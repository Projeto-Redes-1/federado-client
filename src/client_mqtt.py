import os
import time
import pickle
import paho.mqtt.client as mqtt
from client import Client  # seu código existente
from logger_utils import setup_logger, get_host_info

CLIENT_ID = int(os.getenv("CLIENT_ID", 0))
BROKER_IP = os.getenv("BROKER_IP", "192.168.1.100")
PARAMS_TOPIC = f"fed/client/{CLIENT_ID}/params"
GLOBAL_TOPIC = "fed/global/params"

logger = setup_logger(f"client_{CLIENT_ID}", f"logs/client_{CLIENT_ID}.log")
hostname, ip = get_host_info()
logger.info(f"🚀 Cliente {CLIENT_ID} iniciado em {hostname} ({ip})")

def on_message(client, userdata, msg):
    logger.info("📥 Parâmetros globais recebidos.")
    with open("global_parameters.pkl", "wb") as f:
        f.write(msg.payload)
    client.disconnect()

mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
mqtt_client.connect(BROKER_IP, 1883, 60)
mqtt_client.subscribe(GLOBAL_TOPIC)
mqtt_client.loop_start()

# Aguarda parâmetros globais iniciais
while not os.path.exists("global_parameters.pkl"):
    logger.info("⏳ Aguardando parâmetros globais...")
    time.sleep(2)

with open("global_parameters.pkl", "rb") as f:
    parameters = pickle.load(f)

start_train = time.time()
client = Client(CLIENT_ID)
updated_parameters = client.train(parameters)
duration = time.time() - start_train
logger.info(f"🎓 Treinamento local concluído em {duration:.2f} segundos.")

payload = pickle.dumps(updated_parameters)
mqtt_client.publish(PARAMS_TOPIC, payload)
logger.info(f"📤 Enviou {len(payload)} bytes para o servidor via MQTT.")
mqtt_client.loop_stop()
logger.info("🏁 Cliente finalizado.")
