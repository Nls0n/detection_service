from ultralytics import YOLO


def train_yolov8():

    model = YOLO('WS/exp9/weights/last.pt')

    params = {

        'data': 'data.yaml',
        'epochs': 25,
        'batch': 6,
        'imgsz': 1280,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'weight_decay': 0.0005,

        'box': 1.0,
        'cls': 0.1,
        'dfl': 0.05,
        'kobj': 1.4,

        'conf': 0.05,
        'iou': 0.65,
        'multi_scale': True,

        'augment': True,
        'mosaic': 1.0,
        'copy_paste': 0.8,
        'mixup': 0.3,
        'hsv_h': 0.02,
        'hsv_s': 0.9,
        'hsv_v': 0.6,
        'flipud': 0.0,
        'fliplr': 0.0,
        'degrees': 14.0,
        'translate': 0.1,

        'erasing': 0.5,

        'warmup_epochs': 5,
        'patience': 15,
        'close_mosaic': 10,

        'project': 'WS',
        'name': 'exp10'
    }

    model.train(**params)

if __name__ == '__main__':
    train_yolov8()