import paho.mqtt.client as mqtt
import time
import json
import subprocess

# MQTT Broker/topic
BROKER = '192.168.1.20'
PORT = 1883
TOPIC = 'O2O/SensorControl'


# パスワードの入力
passwd = ('a\n').encode()

# Callback function
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
        client.subscribe(TOPIC)
    else:
        print("Connection failed with code", rc)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload)
        if payload.get("command") == 1:
            print("shutdown now!")
            subprocess.run(["sudo", "shutdown", "-h", "now"], input=passwd)
        elif payload.get("command") == 2:
            print("reboot now!")
            subprocess.run(["sudo", "shutdown", "-r", "now"], input=passwd)
        else:
            print(f"Message received: {msg.topic} {msg.payload}")
    except json.JSONDecodeError:
        print("Received non-JSON message")


while True:
    try:
        # make client
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message

        # connect broker
        while True:
            try:
                client.connect(BROKER, PORT, 60)
                break
            except Exception as e:
                print(f"Waiting for broker to start... {e}")
                time.sleep(1)

        # MQTT loop
        client.loop_forever()
    except:
        print("program error")
        time.sleep(1)
