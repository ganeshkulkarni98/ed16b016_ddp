import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate

import utils
import transforms as T
from datasets import CustomCocoDataset

def model_init(model_name):
  
    if model_name == 'FasterRCNN':

      weight_file_path = '/content/FasterRCNN/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
      #weight_file_path = '/content/FasterRCNN/CP_epoch2.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating model...")
    '''
    torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, 
    num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs)

    '''

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = False, progress = True, 
                  num_classes = num_classes, pretrained_backbone = False)

    print('Loading weight file...')
    model.load_state_dict(torch.load(weight_file_path, map_location=device))

    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model.to(device)
    print('model initialized')

    return model, device

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():

    print("Creating data loaders")

    test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size= test_batch_size,
        sampler=test_sampler, num_workers=workers,
        collate_fn=utils.collate_fn)

    model, device = model_init(model_name) 
    model.to(device) 
    model.eval()

    evaluate(model, test_data_loader, device=device)

if __name__ == "__main__":

    # Hyper parameters
    workers = 4              # number of data loading workers
    print_freq = 4

    # inputs 

    test_image_folder = '/content/data/val2017'
    test_coco_dataset = '/content/data/test_coco_dataset.json'

    test_batch_size = 2

    model_name = 'FasterRCNN'

    # Data loading code
    print("Loading data")

    test_dataset = CustomCocoDataset(test_image_folder, test_coco_dataset,  get_transform(train=False))

    #NAME = test_dataset.get_names()

    NAME = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter',
      'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie',
      'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
      'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 
      'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 
      'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    num_classes = len(NAME)

    # Run test function
    main() 
