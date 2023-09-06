import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import logging
import os
import argparse
def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--save_folder",
            type=str,
            default='',
            help="The folder, which will contain created images.",
        )
        parser.add_argument(
            "--label",
            type=str,
            required=True,
            help="label of sign language",
        )
        parser.add_argument(
            "--num_imgs",
            type = int,
            default = 10000,
            help="number of images will be created",
        )
        parsed_args = parser.parse_args()

        return parsed_args

def synthesis_Data(args):
        cap = cv2.VideoCapture(0)
        detector = HandDetector(maxHands=1)

        offset = 40
        imgSize = 300
        if not args.save_folder:
            args.save_folder = args.label
        os.makedirs(args.save_folder, exist_ok=True)
        counter = 0
        while counter<args.num_imgs:
            try:
                if counter <30: 
                    counter += 1
                    continue 
                counter += 1
                success, img = cap.read()
                copy_img = img.copy()
                hands, img = detector.findHands(img)
                h_img, w_img, _ = img.shape			
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']
                    imgCrop = copy_img[max(y - offset, 0):min(y + h + offset, h_img), max(0, x - offset):min(w_img, x + w + offset)]
                    cv2.imwrite(f'{args.save_folder}/Image_{time.time()}.jpg',imgCrop)
                cv2.imshow("Image", img)
                key = cv2.waitKey(1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('brek')
                    break
            except Exception as e: # work on python 3.x
                print('Failed to upload to ftp: '+ str(e))
if __name__ == '__main__':
        args = parse_arguments()
        synthesis_Data(args)
