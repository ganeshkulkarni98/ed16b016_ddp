# Train and Predict using FasterRCNN on Custom Dataset in pytorch 

This contains reference training scripts for object detection.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

Except otherwise noted, all models have been trained on 8x V100 GPUs.



### Training and Testing of Faster R-CNN
```
python train.py
```
In order to train FasterRCNN, input custom dataset should be in COCO dataset format.
Sample of custom dataset for understading is given on this github.
JSON file of dataset should be in COCO dataset format
format of JSON file should be like, {'images': [], 'categories': [], 'annotations': []}

create output directory to save models stats at each epoch

### Only Testing of Faster R-CNN
```
python test.py
```
This line of code it to test model only.

### Prediction using Faster R-CNN
```
python predict.py
```
After training, prediction is done using .pth weight file
Create directory to save save input images having predicted boundry boxes on it.
