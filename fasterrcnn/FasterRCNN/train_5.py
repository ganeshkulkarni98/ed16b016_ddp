r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
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
#import transforms as T1
from torchvision import transforms
from datasets import CustomCocoDataset
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def model_init(model_name):
  
    if model_name == 'FasterRCNN':

      weight_file_path = '/lfs/usrhome/btech/ed16b016/scratch/project/fasterrcnn/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating model...")
    '''
    torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, 
    num_classes=91, pretrained_backbone=True, trainable_backbone_layers=3, **kwargs)

    '''

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = False, progress = True, 
                                    pretrained_backbone = False)

    print('Loading weight file...')
    model.load_state_dict(torch.load(weight_file_path, map_location=device))

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model.to(device)
    print('model initialized')

    return model, device


def get_transform(train):
    T = []
    
    #transforms.append(T.Scale(size = 640))
    
    T.append(transforms.Resize((640,640),interpolation=Image.NEAREST))
    T.append(transforms.ToTensor())
    
    #if train:
    #    transforms.append(T.RandomHorizontalFlip(0.5))
    return transforms.Compose(T)


def main():

    print("Creating data loaders")

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(train_dataset, k=aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, train_batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, train_batch_size, drop_last=True)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, num_workers=workers,
        collate_fn=utils.collate_fn)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size= val_batch_size,
        sampler=test_sampler, num_workers=workers,
        collate_fn=utils.collate_fn)

    model, device = model_init(model_name)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)

    if resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    for epoch in range(0, epochs):

        train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq)
        lr_scheduler.step()

        # evaluate after every epoch
        evaluate(model, val_data_loader, device=device)
        
        torch.save(model.state_dict(), f'/lfs/usrhome/btech/ed16b016/scratch/project/fasterrcnn/FasterRCNN/results/CP_epoch{epoch + 1}.pth')
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":

    # Hyper parameters
    epochs = 50
    workers = 8              # number of data loading workers
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0005
    lr_step_size = 8
    lr_steps = [16, 22]
    lr_gamma = 0.1
    print_freq = 4
    #output_dir =
    resume = False
    start_epoch = 0
    aspect_ratio_group_factor = 3
    world_size = 1                 # number of distributed processes
    dist_url = 'env://'            #url used to set up distributed training

    # inputs 

    train_image_folder = '/lfs/usrhome/btech/ed16b016/scratch/project/yolo/train_png_div_2'
    train_coco_dataset = '/lfs/usrhome/btech/ed16b016/scratch/project/fasterrcnn/train_4.json'

    val_image_folder = '/lfs/usrhome/btech/ed16b016/scratch/project/yolo/train_png_div_2'
    val_coco_dataset = '/lfs/usrhome/btech/ed16b016/scratch/project/fasterrcnn/test_4.json'

    train_batch_size = 40
    val_batch_size = 40

    model_name = 'FasterRCNN'

    # Data loading code
    print("Loading data")

    
    train_dataset = CustomCocoDataset(train_image_folder, train_coco_dataset,  get_transform(train=True))
    val_dataset = CustomCocoDataset(val_image_folder, val_coco_dataset,  get_transform(train=False))

    #NAME = train_dataset.get_names()

    NAME = ['__background__',  'Aortic enlargement', 'Atelectasis', 
    'Calcification', 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity', 
    'Nodule/Mass', 'Other lesion', 
     'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis', 'No Findings']
    
    num_classes = len(NAME)

    # Run train function
    main() 
