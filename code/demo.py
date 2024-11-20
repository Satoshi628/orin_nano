#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import os
import argparse

# third party library
import pyrealsense2 as rs
import numpy as np
import cv2
import onnxruntime

# my library
from utils.process import RGBDCamera, HumanInfoExtractor
from utils.SpeedTester import SpeedTester
from utils.mqtt import MQTT_Pub


if __name__ == "__main__":
    camera = RGBDCamera("D435", varpose=True)
    model = HumanInfoExtractor("models/movenet_multipose_lightning_1.onnx",
                                "D435", #D435 or D455
                                camera_pos_y=1.0)
    broker = 'localhost'
    port = 1883
    topic = "python/mqtt"
    # mqtt = MQTT_Pub(broker, port, topic)

    for rgb_image, d_image in camera():
        outputs = model(rgb_image, d_image)

        camera.update(outputs["keypoints"], outputs["keypoints_Real"], outputs["bbox_Real"], outputs["normal_vectors"], outputs["status"])


        for key, value in outputs.items():
            if isinstance(value, np.ndarray):
                outputs[key] = value.tolist()
        # mqtt.publish(outputs)
