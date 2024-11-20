#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import os
import argparse
from time import sleep
import time
# third party library
import numpy as np
import cv2
import onnxruntime
import pyrealsense2 as rs

# my library
from utils.process import RGBDCamera, HumanInfoExtractor
from utils.SpeedTester import SpeedTester
from utils.mqtt import MQTT_Pub


Error_path = "/home/rb53262/Desktop/orin_nano/code/test.txt"

def camera_init():
    while True:
        try:
            camera = RGBDCamera(varpose=False)
            break
        except:
            with open(Error_path, mode="w") as f:
                f.write("Realsense_Conect_Error")
            print("Realsense_Conect_Error")
            sleep(1)
    return camera


def mqtt_init(broker='192.168.1.199', port=1883, topic="O2O/ObjectDetectionData"):
    while True:
        try:
            mqtt = MQTT_Pub(broker, port, topic)
            break
        except:
            with open(Error_path, mode="w") as f:
                f.write("MQTT_Error")
            print("MQTT_Error")
            sleep(1)
    return mqtt


if __name__ == "__main__":
    while True:
        try:
            camera = camera_init()
            center_xy, F_xy = camera.get_intrinsics()
            model_path = "/home/rb53262/Desktop/orin_nano/code/pint_models/yolov9_n_discrete_headpose_post_0100_1x3x480x640.onnx"
            model = HumanInfoExtractor(model_path,
                                        center_xy=center_xy,
                                        F_xy=F_xy,
                                        bbox_score_th=0.5,
                                        camera_pos_y=0.86)
            
            mqtt = mqtt_init(broker = '192.168.1.20', port = 1883, topic = "O2O/ObjectDetectionData")
            
            model.model(image=np.zeros([480,640,3]))
            for rgb_image, d_image in camera():
                bboxes = model(rgb_image, d_image)

                camera.update(bboxes)
                #if len(bboxes):
                #    continue
                outs = {"sensor_name": "DEPTH-SENSOR",
                        "obj_num":len(bboxes),
                        "object":[
                            {
                                "x":bbox.rcx,
                                "y":bbox.rcy,
                                "z":bbox.rcz,
                                "ox":bbox.nx,
                                "oy":bbox.ny,
                                "oz":bbox.nz,
                                "w":0.43,
                                "l":0.43,
                                "h":1.645,
                                "tag_info":idx,
                                "velocity_1d":0.,
                                "pose":bbox.statu,
                                }
                            for idx, bbox in enumerate(bboxes)]}
                # print(outs)
                if len(bboxes):
                    mqtt.publish(outs)
                    pass
        except Exception as e:
            print(e)
            with open(Error_path, mode="w") as f:
                f.write("Process Erro")
            print("Process Error")
            del camera
            sleep(1)
