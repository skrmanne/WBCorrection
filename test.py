#imports
import os, sys
import cv2
import argparse
import numpy as np
import torch
import tqdm

from model import deepWBnet
from utils import *

# create an arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--filename', help="filename.txt which contains filenames to test", default="")
parser.add_argument('--out_folder', help="output base foldername", default="")
parser.add_argument('--weight_path', help="checkpoint path", default="checkpoints/model.pth")
args = parser.parse_args()

# load the checkpoint from model
model = deepWBnet()
checkpoint = torch.load(args.weight_path)
model.load_state_dict(checkpoint) #['model_state_dict'])
model = model.cuda()
model.eval()

filenames = []
with open(args.filename) as f:
    filenames = f.readlines()
    filenames = [x.strip() for x in filenames]

total_mae, total_mse = 0, 0
for filename in tqdm.tqdm(filenames):
    # read an image, label and convert them to tensors.
    label_name = filename.replace("input", "label")
    resized_input_name = filename.replace("input", "input_low")
    output_name = filename.replace("input", "our")
    resized_label_name = filename.replace("input", "label_low")

    lab = cv2.imread(label_name)
    lab = cv2.resize(lab, (256, 256), interpolation=cv2.INTER_AREA)
    #cv2.imwrite(resized_label_name, lab)

    img = cv2.imread(filename)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    #cv2.imwrite(resized_input_name, img)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0

    img = torch.from_numpy(img)
    img = torch.permute(img, (2, 0, 1)).unsqueeze(0) # H,W,C -> 1,C,H,W

    # inference
    img = img.cuda()
    with torch.no_grad():
        out = model(img)

    # save the output
    out = torch.clamp(torch.squeeze(out), 0.0, 1.0)*255
    out = torch.permute(out, (1, 2, 0)).cpu().numpy()
    out = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)
    #cv2.imwrite(output_name, out)

    # evaluation
    total_mae, total_mse = total_mae+calc_mae(out, lab), total_mse+calc_mse(out, lab)

print("Mean Angular Error: ", total_mae/len(filenames))
print("Mean Squared Error: ", total_mse/len(filenames))
