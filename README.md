# ed16b016_ddp
Final year project on the Chest X-Rays Abnormilities Detection 


I have implemented two state-of-the-art Object Detection deep learning techniques.

1) Yolo
2) Faster RCNN

I have used the aqua cluster setup to access the GPU. For that, cmd files have been created and added to run the corresponding python script.

## Detectron

I have used Faster RCNN using detection environment as it contains more features to tune the hyperparameters. In the detection folder, I used the COCO format dataset for VinBig and used it as an input dataset.

## Doc

It consists the managing relations between different files and recording the changes and corresponding results and outcomes.

## Faster RCNN

I have also used Faster RCNN in the PyTorch environment. In this Faster RCNN folder, the COCO format is used and trained the model.

## Kfold_yolo

It consists of the K fold Yolo method and Stage 2 k fold Yolo method.

### part_zfturbo

This folder consists of preprocessing the dataset, implementing the yolov5 model, and generating the file for submission.

### part_sergey

This folder consists of the stage 2 K fold method, preprocessing the dataset further, and generating the dataset.

## Output

This folder consists of the results of the models.

## preprocess data

This folder consists of pythons scripts to convert the dataset into COCO format, Yolo format, generate the results, preprocess the dataset, and others.

## Result

This folder consists of the output format results for the VinBig dataset.

## yolo

This folder consists of the yolov5 scripts to run the model for single model-based training with you. 
