# dataloader
import cv2
import numpy as np
from torch.utils.data import Dataset

class AWBDataset(Dataset):
    """White balance correction post camera pipeline.
    Dataset contains images with incorrect white-balance and corresponding GT image.
    """
    def __init__(self, filename, transform=None):
        with open(filename) as f:
            self.filenames = f.readlines()
            self.filenames = [x.strip() for x in self.filenames]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        input_fname = self.filenames[idx]
        label_fname = input_fname.replace("input", "label") #TODO based on folder structure.

        # read image and label
        img = cv2.imread(input_fname)
        lab = cv2.imread(label_fname)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        lab = cv2.cvtColor(lab, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

        # albumentation augmentations handles train/test variations and applies augs
        # consistently to image and label
        aug = self.transform(image=img, label=lab)
        img, lab = aug['image'], aug['label']

        return img, lab
