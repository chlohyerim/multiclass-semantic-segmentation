import cv2
import os
from tqdm import tqdm

if __name__ == '__main__':
    src_dir = 'data/test/img'
    dst_dir = src_dir

    for filename in tqdm(os.listdir(src_dir)):
        if (filename.endswith('.png') or filename.endswith('.PNG')):
            img = cv2.imread(os.path.join(src_dir, filename))

            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

            img_processed = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

            cv2.imwrite(dst_dir + '/' + os.path.splitext(filename)[0] + '.PNG', img_processed)