import os
import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from PIL import Image
import json
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import utils
from datasets import *

def get_prediction(model, device, img_path, threshold, NAMES):
  
  img = Image.open(img_path) # Load the image

  img_transforms = transforms.Compose([transforms.ToTensor()])
  img = img_transforms(img)
  pred = model([img.to(device)]) # Pass the image to the model
  pred_class = [NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())] # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().cpu().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  pred_score = pred_score[:pred_t+1]
  return pred_boxes, pred_class, pred_score

def object_detection_api(model, device, img_path, threshold, rect_th, text_size, text_th, NAMES, output_image_path):
  
  boxes, pred_cls, pred_score = get_prediction(model, device, img_path, threshold, NAMES) # Get predictions
  img = cv2.imread(img_path) # Read image with cv2
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
  img_result = []
  for i in range(len(boxes)):
    obj ={}
    obj['xy_top_left'] = boxes[i][0]
    obj['xy_bot_right'] = boxes[i][1]
    obj['conf_level'] = pred_score[i]
    obj['label']= pred_cls[i]
    img_result.append(obj)
    cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
    cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
    
  cv2.imwrite(output_image_path, img)
  return img, img_result

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

def label_map_fn(json_file_path):
    anno = json.load(open(json_file_path))
    categories = anno["categories"]
    labels = ['__background__']
    for i in range (0,len(categories)):
      labels.append(str(categories[i]["name"]))

    return labels

def main():

  model, device = model_init(model_name)

  model.to(device)
  
  model.eval()
  
  imgs = list(sorted(os.listdir(test_image_folder)))

  results = []
  images = []
  for img in imgs:
    img_path = os.path.join(test_image_folder, img)
    output_image_path = os.path.join('/content/results', img)
    image, img_result = object_detection_api(model, device, img_path, threshold, rect_th, text_size, text_th, NAME, output_image_path)
    images.append(image)
    results.append(img_result)
  
  print(images, results)

if __name__ == "__main__":

    # Hyper parameters
    threshold = 0.6       #threshold value
    rect_th = 1            #rect_th value
    text_size = 0.5
    text_th = 1

    # inputs 

    test_image_folder = '/content/data/test_images'
    #test_coco_dataset = '/content/data/test_coco_dataset.json'

    model_name = 'FasterRCNN'

    #NAME = label_map_fn(test_coco_dataset)

    NAME = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter',
      'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie',
      'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
      'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 
      'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 
      'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    num_classes = len(NAME)

    # Run predict function
    main()
