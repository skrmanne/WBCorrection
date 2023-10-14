# imports
import os, sys
import argparse
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import AWBDataset
from model import deepWBnet

# args
parser = argparse.ArgumentParser(description='White balance correction')
parser.add_argument("--batch_size", type=int, help="Batch size", default=8)
parser.add_argument("--lr", type=float, help="LR", default=1e-4)
parser.add_argument("--epochs", type=int, help="Num epochs", default=50)
parser.add_argument("--weight_path", type=str, help="Checkpoint store path", default="checkpoints/awb")
args = parser.parse_args()

# create train and test transforms
test_transform = A.Compose([
    A.Resize(256, 256),
    ToTensorV2(),
    ],
    additional_targets={'image': 'image', 'label': 'image'})

train_transform = A.Compose([
    A.RandomResizedCrop(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ToTensorV2(),
    ],
    additional_targets={'image': 'image', 'label': 'image'})

# create train and test dataloaders
train_dataset = AWBDataset("data/train.txt", train_transform)
test_dataset = AWBDataset("data/test.txt", test_transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

# create a model
model = deepWBnet()
model.load_state_dict(checkpoint) #['model_state_dict'])
model = model.to('cuda') # move to GPU

# train and test
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

log_dir = "logs"  # The directory where log files will be saved
writer = SummaryWriter(log_dir)

best_err = 1e8
for epoch in range(args.epochs):
    # set model to train
    model.train()

    tloss, vloss, fmae = 0.0, 0.0, 0.0
    for idx, data in enumerate(train_loader):
        img, gt = data
        # move to device
        img, gt = img.cuda(), gt.cuda()

        #print("img stats:", torch.min(img), torch.max(img))
        #print("gt stats:", torch.min(gt), torch.max(gt))
        optimizer.zero_grad()
        out = model(img)

        # loss:
        loss = loss_fn(out, gt)
        loss.backward()

        optimizer.step()
        tloss += loss.item()

    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            img, gt = data
            # move to device
            img, gt = img.cuda(), gt.cuda()

            out = model(img)
            # loss 
            loss = loss_fn(out, gt)
            vloss += loss.item()
    
    tloss, vloss = tloss/len(train_loader), vloss/len(test_loader)
    if vloss <= best_err:
        best_err = vloss

        # save the model as checkpoint
        torch.save(model.state_dict(), os.path.join(args.weight_path, "weights.pth"))

    # Log the losses
    writer.add_scalar('Loss/Train', tloss, epoch)
    writer.add_scalar('Loss/Validation', vloss, epoch)
    print("Epoch:{epoch} : train loss: {tloss}, validation loss: {vloss}:".format(
        epoch=epoch, tloss=tloss, vloss=vloss))

# close the summary writer
writer.close()