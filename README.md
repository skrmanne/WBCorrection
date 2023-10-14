# Correcting an incorrectly white-balanced image

Computational color constancy in computer vision is the task of normalizing the effect of an illuminant in the scene to predict the colors of the objects when imaged under normal (white) lighting. This problem, however, assumes the input image data available is in linear domain making the illuminant estimation simpler by decomposing the image into radiance and illumination. In this project, we are interested in a slightly different problem – what if the data available is a processed image from the camera pipeline instead of linear data straight from the sensor? Such cases can occur when the camera’s auto white balance fails or when taking pictures in manual mode and setting an incorrect white-balance setting. Since the image is now a product of incorrect white balancing followed by a series of non-linear operations recovering accurate white-balanced image becomes challenging. We test a simple neural network on images captured from smartphones and a DSLR.

## Installation
The required packages can be easily installed through a conda environment as follows:

```conda create --name <env> --file requirements.txt```

## Dataset
We use Set-2 from [here](https://cvil.eecs.yorku.ca/projects/public_html/sRGB_WB_correction/dataset.html). Place the dataset (images and gruond-truth in a folder) as specified below:
```
Dataset root:
    |- input
        |- abc.png
        |- lmn.png
        ...
        |- xyz.png
    |- label
        |- abc.png
        |- lmn.png
        ...
        |- xyz.png
```

Please change the dataset roots in 'test.txt' and 'train.txt' splits in `data/` according to the downloaded path.


## Training and Testing
Please use the sample train and test commands once the data is setup as described above:

### Train:
```python train.py```

### Test:
```python test.py --filename data/test.txt --weight_path checkpoints/model.pth ```