import numpy as np
import cv2

# Calculates mean angular error between two images.
def calc_mae(source, target):
  source = np.reshape(source, [-1, 3]).astype(np.float32)
  target = np.reshape(target, [-1, 3]).astype(np.float32)

  source_norm = np.sqrt(np.sum(np.power(source, 2), 1))
  target_norm = np.sqrt(np.sum(np.power(target, 2), 1))
  norm = source_norm * target_norm
  
  L = np.shape(norm)[0]
  inds = norm != 0
  angles = np.sum(source[inds, :] * target[inds, :], 1) / norm[inds]
  angles[angles > 1] = 1
  
  f = np.arccos(angles)
  f[np.isnan(f)] = 0
  f = f * 180 / np.pi
  
  return sum(f)/L

# pixel wise mse.
def calc_mse(source, target):
  source = np.reshape(source, [-1, 1]).astype(np.float64)
  target = np.reshape(target, [-1, 1]).astype(np.float64)
  mse = sum(np.power((source - target), 2))

  return mse / ((np.shape(source)[0]))


# calculate average MAE and MSE for a model
def calc_metrics(output_folder, label_folder):
    mae, mse = 0, 0

    # expects same filenames in both
    for fname in os.listdir(output_folder):
        output_fname = os.path.join(output_folder, fname)
        label_fname = os.path.join(label_folder, fname)

        output = cv2.imread(output_fname)
        label = cv2.imread(label_fname)

        mae += calc_mae(output, label)
        mse += calc_mse(output, label)
    
    print("Mean Angular Error: ", mae/len(os.listdir(output_folder)))
    print("Mean Squared Error: ", mse/len(os.listdir(output_folder)))

