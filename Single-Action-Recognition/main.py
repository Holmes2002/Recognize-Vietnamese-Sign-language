import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import logging
import os
import argparse
from inference import predict_label, load_model
def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--save_checkpoint",
            type=str,
            default='',
            help="The folder, which will contain created images.",
        )
        parser.add_argument(
            "--labels",
            type=str,
            required=True,
            help="labels of sign language (txt)",
        )
        parsed_args = parser.parse_args()

        return parsed_args

def Inference(args):
        cap = cv2.VideoCapture(0)
        detector = HandDetector(maxHands=1)
        labels = open(args.labels, 'r').read().splitlines()
        model = load_model(args.save_checkpoint, len(labels))
        offset = 40
        imgSize = 300
        while True:
                success, img = cap.read()
                copy_img = img.copy()
                hands, img = detector.findHands(img)
                h_img, w_img, _ = img.shape			
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']
                    imgCrop = copy_img[max(y - offset, 0):min(y + h + offset, h_img), max(0, x - offset):min(w_img, x + w + offset)]

                    predict = model(imgCrop)
                    cv2.putText(copy_img, labels[predict], (max(0, x - offset-3), max(y - offset-3, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('brek')
                    break
if __name__ == '__main__':
        Inference(*args)
