import pycocotools
import cv2
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import numpy as np
from detectron2.structures import BoxMode
import detectron2
from detectron2.config.config import CfgNode as CN
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
import gc
setup_logger()


def get_celi_data(dir_path):

    images = os.listdir(dir_path)

    dataset_dict = []
    idx = 0
    for image in images:
        print(idx)
        record = {}

        filename = os.path.join(dir_path, image, 'images', image + '.png')
        print(filename) 
        img = cv2.imread(filename)
        height, width = img.shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        masks = os.listdir(os.path.join(dir_path, image, 'masks'))
        print(masks)
        for mask in masks:
            maskname = os.path.join(dir_path, image, 'masks', mask)
            mask_img = cv2.imread(maskname, 0)
            masknp = np.asarray(mask_img)

            pos = np.where(masknp > 0)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            seg = pycocotools.mask.encode(np.asarray(masknp, order="F"))

            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": seg,
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dict.append(record)
        idx += 1
    return dataset_dict


if __name__ == '__main__':
    for d in ["train", "val"]:
        DatasetCatalog.register("ceil_" + d, lambda d=d: get_celi_data("dataset/" + d))
        MetadataCatalog.get("ceil_" + d).set(thing_classes=["ceil"])
    ceil_metadata = MetadataCatalog.get("ceil_train")

    cfg = get_cfg()
    print(cfg)
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("ceil_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = os.path.join("./input_model", "best.pth")
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.SOLVER.STEPS = []

    
    cfg.SOLVER.BASE_LR = 0.001  
    epoch = 10
    cfg.SOLVER.MAX_ITER = epoch * 240
    cfg.SOLVER.MAX_ITER = 2400

    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 2
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 50
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 50
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 50
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 50

    cfg.MODEL.RPN.NMS_THRESH = 0.8

    
    cfg.INPUT.CROP = CN({"ENABLED": True})
    cfg.INPUT.CROP.TYPE = "absolute"
    cfg.INPUT.CROP.SIZE = [600, 600]

    

    cfg.OUTPUT_DIR = "./output_model"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()
    