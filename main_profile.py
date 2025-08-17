import os
import warnings
warnings.filterwarnings('ignore')
import torch
from ultralytics import RTDETR
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("ultralytics/cfg/models/11/yolo11l-ours.yaml")
    # model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-l.yaml")


    model.model.eval()
    model.info(detailed=True)
    try:
        model.profile(imgsz=[640, 640])
    except Exception as e:
        print(e)
        pass
    print('after fuse:', end='')
    model.fuse()