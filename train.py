import warnings

import torch

warnings.filterwarnings('ignore')
from ultralytics import RTDETR
from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    # model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-r18-ours.yaml")
    model = YOLO("ultralytics/cfg/models/11/yolo11-ours.yaml")
    model.train(data='ultralytics/cfg/datasets/coco8.yaml',
                nwdloss= True,
                #cache=True,#
                imgsz=640,
                epochs=300,
                batch=1,#调小，fix  可能会因为显存不足而报错,-1也可能会报错out of memory
                workers=0,#fix 页面太小，可能会报错
                device='0',
		        # amp=True,
                # time=0.01,
                # resume='runs/train/YOLOv8-TEMP_CAFM_v210/weights/last.pt', # last.pt path
                project='runs/train',
                name='yolov8',
                # save=True,
                # save_period=1,#每个epoch保存一次
                # resume=True,
                cfg='ultralytics/cfg/yolov8n-visdrone.yaml'
                )