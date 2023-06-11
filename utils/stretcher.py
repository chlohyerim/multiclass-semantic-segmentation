import cv2
import os
from tqdm import tqdm

if __name__ == '__main__':
    src_dir = 'data/train/gt'
    dst_dir = src_dir

    for filename in tqdm(os.listdir(src_dir)):
        if (filename.endswith('.PNG') or filename.endswith('.png') or filename.endswith('.JPG') or filename.endswith('.jpg')):
            img = cv2.imread(os.path.join(src_dir, filename))

            img_processed = cv2.resize(img, dsize=(480, 368), interpolation=cv2.INTER_AREA)

            cv2.imwrite(dst_dir + '/' + os.path.splitext(filename)[0] + '.PNG', img_processed)