import warnings

import torch

warnings.filterwarnings('ignore')
from ultralytics import RTDETR,YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\py\results\update-s\best.pt') # select your model.pt path

    model.predict(source=r'D:\py\datasets\visdrone_detect',
                  conf=0.25,
                  project='runs/detect',
                  name='update-l,best',
                  save=True,
                #   visualize=True # visualize model features maps
                  )