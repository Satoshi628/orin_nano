#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
from __future__ import annotations
import os
import re
import time
import json

# third party library
from paho.mqtt import client as mqtt_client

# my library

class MQTT_Pub():
    def __init__(self, broker: str,port: int, topic: str) -> None:
        self.broker = broker
        self.port = port
        self.topic = topic
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker!")
            else:
                print("Failed to connect, return code %d\n", rc)

        self.client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION1)
        self.client.on_connect = on_connect
        self.client.connect(self.broker, self.port)

    def publish(self, data: dict) -> None:
        payload = json.dumps(data)
        result = self.client.publish(self.topic, payload)
        status = result[0]
        if status != 0:
            print(f"Failed to send message to topic {topic}")


class MQTT_Sub():
    def __init__(self, broker: str,port: int, topic: str) -> None:
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = mqtt_client.Client()
        self.received_data = None  # 受信データを保持する属性
        
        # コールバック関数の設定
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        """MQTTブローカーに接続する"""
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_start()  # メッセージの送受信を非同期で開始

    def on_connect(self, client, userdata, flags, rc):
        """接続が成功したときの処理"""
        if rc == 0:
            print("Connected to broker")
            # トピックを購読
            self.client.subscribe(self.topic)
        else:
            print(f"Connection failed with code {rc}")

    def on_message(self, client, userdata, msg):
        """メッセージを受信したときの処理"""
        try:
            # 受信したJSONデータをPythonのデータ型に変換
            self.received_data = json.loads(msg.payload.decode())
            print(f"Received message: {self.received_data}")
        except json.JSONDecodeError:
            print("Failed to decode JSON data")

    def get_received_data(self):
        """受信したデータを取得"""
        return self.received_data

    def disconnect(self):
        """接続を終了する"""
        self.client.loop_stop()
        self.client.disconnect()


if __name__ == "__main__":
    broker = 'localhost'
    port = 1883
    topic = "python/mqtt"

    mq_pub = MQTT_Pub(broker, port, topic)
    mq_sub = MQTT_Sub(broker, port, topic)
    data = {
        "sequence1": [1, 2, 3, 4, 5],
        "sequence2": [6, 7, 8, 9, 10],
        "temperature": 25.5,
        "status": "active"
    }

    # メッセージを受信するまで待機
    import time
    mq_pub.publish(data)
    time.sleep(3)  # この間にメッセージを受信する

    # 受信したデータを取得
    received_data = mq_sub.get_received_data()
    print(f"Returned received data: {received_data}")

    # 接続を終了
    mq_sub.disconnect()
