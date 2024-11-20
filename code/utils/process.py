#!/usr/bin/env python
# -*- coding: utf-8 -*-

# standard library
import os
import argparse
import re
import time
import random
import math

# third party library
import glob
import numpy as np
import pyrealsense2 as rs
import cv2
import onnxruntime

# my library
from .HumanPosePram import (HEADDIRECTION_NAME,
                            UNKNOWN,
                            Standing,
                            Sitting,
                            Decubitus,
                            NORMAL_VECTORS,
                            STATUS_NAME2IDX,
                            STATUS_IDX2NAME,
                            SITUATION_COLOR,
                            NON_DATA)
from .pinto import YOLOv9


class RGBDCamera():
    def __init__(self, WH=[640, 360], fps=30, varpose=True):
        self.varpose = varpose

        config = rs.config()
        # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, WH[0], WH[1], rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, WH[0], WH[1], rs.format.z16, fps)

        # ストリーミング開始
        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(config)

        # Alignオブジェクト生成
        self.align = rs.align(rs.stream.color)
        self.hole_filling = rs.hole_filling_filter()

        self.bboxes = None
        self.fps = 0.0
        

    def __call__(self):
        fps = 0.0
        while True:

            # フレーム待ち(Color & Depth)
            frames = self.pipeline.wait_for_frames()

            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            depth_frame = self.hole_filling.process(depth_frame).as_depth_frame()
            if not depth_frame or not color_frame:
                continue

            # imageをnumpy arrayに
            color_image = np.asanyarray(color_frame.get_data())
            depth_image_np = np.asanyarray(depth_frame.get_data())

            # depth imageをカラーマップに変換
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_np, alpha=0.08), cv2.COLORMAP_JET)

            # 画像表示
            if self.varpose:
                draw_image = self._draw(color_image, self.bboxes)
                cv2.putText(depth_colormap, f'FPS:{self.fps:.2f}', (540, 350), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
                images = np.hstack((draw_image, depth_colormap))
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                if cv2.waitKey(1) & 0xff == 27:  # ESCで終了
                    cv2.destroyAllWindows()
                    break
            
            yield color_image, depth_frame
        # ストリーミング停止
        self.pipeline.stop()
        yield color_image, depth_frame

    def _draw(self, image, bboxes):
        image = image.copy()
        if bboxes is not None:
            for bbox in bboxes:
                #bbox描写
                cv2.rectangle(image, [bbox.x1, bbox.y1], [bbox.x2, bbox.y2], color=(0,0,255), thickness=3)
                
                #姿勢カラーを設定
                color = SITUATION_COLOR[STATUS_IDX2NAME[bbox.statu]]
                #statu描写
                image = cv2.putText(image, f"{HEADDIRECTION_NAME[bbox.headdirection]}-{STATUS_IDX2NAME[bbox.statu]}", [bbox.cx, bbox.cy], cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)
                #座標描写
                image = cv2.putText(image, f"[{bbox.rcx:.2f},{bbox.rcy:.2f},{bbox.rcz:.2f}]", [bbox.cx, bbox.cy+20], cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)
                image = cv2.putText(image, f"[{bbox.nx:.2f},{bbox.ny:.2f},{bbox.nz:.2f}]", [bbox.cx, bbox.cy+40], cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)
        return image

    def update(self, bboxes, fps=0.0):
        self.bboxes = bboxes
        self.fps = fps

    def get_intrinsics(self):
        intr = rs.video_stream_profile(self.profile.get_stream(rs.stream.color)).get_intrinsics()
        center_xy = [intr.ppx, intr.ppy]
        F_xy = [intr.fx, intr.fy]
        return center_xy, F_xy

class HumanInfoExtractor():
    def __init__(self, model_path, center_xy, F_xy, bbox_score_th=0.2, camera_pos_y=0.8):
        self.model = YOLOv9(
            runtime="onnx",
            model_path=model_path,
            obj_class_score_th=bbox_score_th,
            attr_class_score_th=0.0,
            providers=[
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ],
    )
        self.camera_pos_y = camera_pos_y

        self.center_xy = center_xy
        self.F_xy = F_xy

    def _convert_uint2meter(self, bboxes, depth_frame):
        for bbox in bboxes:
            bbox.rcz = depth_frame.get_distance(int(bbox.cx), int(bbox.cy))
        return bboxes

    def _convert_ImageXY2RealXY(self, bboxes):
        """convert to Image position from Real position

        Args:
            Image_XY (ndarray[..., 2]): Image XY
            depth (ndarray[..., 1]): Depth
            Center_xy (list): image center position.
            F_xy (list): foucus.
        Returns:
            ndarray[..., 2]: Real world position
        """

        for bbox in bboxes:
            bbox.rcx = (bbox.cx - self.center_xy[0]) * bbox.rcz / self.F_xy[0]
            bbox.rcy = (bbox.cy - self.center_xy[1]) * bbox.rcz / self.F_xy[1]
            bbox.rcy *= -1
        
            #頭の距離から水平距離に補正
            bbox.rcz = math.sqrt(abs(bbox.rcz**2 - bbox.rcy**2))
            #水平距離から最小二乗法で求めた誤差を引く
            bbox.rcz -= 0.3491*bbox.rcz - 1.0734
        return bboxes

    def __call__(self, rgb_image, d_image):
        bboxes = self.model(image=rgb_image)

        bboxes = self._convert_uint2meter(bboxes, d_image)
        bboxes = self._convert_ImageXY2RealXY(bboxes)

        # bboxの幅を適当に決める 平均43cm
        for bbox in bboxes:
            bbox.rx1 = bbox.rcx - 0.215
            bbox.ry1 = bbox.rcy - 0.8225
            bbox.rz1 = bbox.rcz - 0.215
            bbox.rx2 = bbox.rcx + 0.215
            bbox.ry2 = bbox.rcy + 0.8225
            bbox.rz2 = bbox.rcz + 0.215
        
        bboxes = self._inference_humanvector(bboxes)
        status = self._inference_situation(bboxes)

        return bboxes

    def _inference_humanvector(self, bboxes):
        for bbox in bboxes:
            vector = np.array(NORMAL_VECTORS[bbox.headdirection])
            if bbox.headdirection != -1: #ノイズを加えない
                vector = vector + np.random.random([3])*0.03
                vector = vector / np.linalg.norm(vector, axis=-1, keepdims=True)
            vector = vector.tolist()
            bbox.nx = vector[0]
            bbox.ny = vector[1]
            bbox.nz = vector[2]
        return bboxes

    def _inference_situation(self, bboxes):
        for bbox in bboxes:
            #鼻の分15cm足す
            if bbox.rcy + 0.15 + self.camera_pos_y > (164.5-1*5.6)/100: # 平均身長：164.5cm±2*5.6cm
                bbox.statu = Standing
            elif bbox.rcy + 0.15 + self.camera_pos_y > (86.97-3*3.1)/100: # 平均座高＋椅子高：86.97cm±2*3.1cm
                bbox.statu = Sitting
            else:
                bbox.statu = Decubitus
        return bboxes




if __name__ == "__main__":
    pass
