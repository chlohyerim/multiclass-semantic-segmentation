import cv2
import numpy as np
import os
from tqdm import tqdm

if __name__ == '__main__':
    src_dir = 'data/train/gt'
    dst_dir = src_dir

    for filename in tqdm(os.listdir(src_dir)):
        if (filename.endswith('.PNG') or filename.endswith('.png') or filename.endswith('.JPG') or filename.endswith('.jpg')):
            img = cv2.imread(os.path.join(src_dir, filename))

            r, g, b = cv2.split(img)
            maxval = np.maximum(np.maximum(r, g), b)

            img_processed = np.zeros_like(img)
            img_processed[maxval == r] = [255, 0, 0]
            img_processed[maxval == g] = [0, 255, 0]
            img_processed[maxval == b] = [0, 0, 255]

            cv2.imwrite(dst_dir + '/' + os.path.splitext(filename)[0] + '.PNG', img_processed)