#coding:utf-8

from ultralytics import YOLO

if __name__ == '__main__':
        model = YOLO('ultralytics/cfg/models/v8/yolov8s+小目标检测头+EMA+ShapeIoU.yaml')
        model.load('yolov8s.pt')  # loading pretrain weights
        model.train(data='mydata.yaml', epochs=500, batch=16, imgsz=640)