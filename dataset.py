import torch
from torch.utils.data import DataLoader

import torchvision.transforms as T

import cv2
import numpy as np
from os import listdir
from os.path import splitext
import random
from tqdm import tqdm

from preprocessor import Preprocessor
import visualizer


# Segmentation을 위한 dataset 클래스 정의
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, gt_dir, is_train):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.is_train = is_train
        self.names = [splitext(file)[0] for file in listdir(img_dir)]


    def __len__(self):
        return len(self.names)
    

    def openimage(self, directory, name):
        return cv2.imread(directory + '/' + name + '.PNG')
    
    
    def __getitem__(self, index):
        name = self.names[index]
        img = self.openimage(self.img_dir, name)
        gt = self.openimage(self.gt_dir, name)
        h, w = img.shape[:2]

        if (self.is_train):
            # Random Horizontal Flip
            r = np.random.random()

            if r < 0.5: 
                img = Preprocessor.flipHorizontal(img)
                gt = Preprocessor.flipHorizontal(gt)

            # Random Perspective
            border = 80
            pts_src = np.float32(
                [[random.randint(0, border), random.randint(0, border)],
                 [w - 1 - random.randint(0, border), random.randint(0, border)],
                 [w - 1 - random.randint(0, border), h - 1 - random.randint(0, border)],
                 [random.randint(0, border), h - 1 - random.randint(0, border)]]
            )

            img = Preprocessor.perspective(img, pts_src, is_gt=False)
            gt = Preprocessor.perspective(gt, pts_src, is_gt=True)

            # Random Rotation(angle in [-60, 60])
            r = np.random.random()
            angle = random.randint(-60, 60)
            scale = np.sqrt(h * h + w * w) / np.min((h, w))
            w_scaled = int(w // 2 * scale) * 2
            h_scaled = int(h // 2 * scale) * 2

            img = Preprocessor.rotate(img, angle, w_scaled, h_scaled, is_gt=False)
            gt = Preprocessor.rotate(gt, angle, w_scaled, h_scaled, is_gt=True)
                
            # Random Resized Crop(scale in [1, 2], cropping at (x, y))
            r = np.random.random()
            
            if r < 0.5:
                scale = random.random() * 1 + 1
                w_scaled = int(w // 2 * scale) * 2
                h_scaled = int(h // 2 * scale) * 2
                x = random.randint(0, np.abs(w_scaled - w))
                y = random.randint(0, np.abs(h_scaled - h))
                
                img = Preprocessor.resizedcrop(img, scale, w_scaled, h_scaled, x, y, is_gt=False)
                gt = Preprocessor.resizedcrop(gt, scale, w_scaled, h_scaled, x, y, is_gt=True)

            # Random Histogram Equalization
            r = np.random.random()

            if r < 0.5:
                img = Preprocessor.equalizeHist(img)

            # Random Reflectance (via homomorphic filter)
            r = np.random.random()

            if r < 0.5:
                gamma2 = 2 ** (np.random.random() * 2 - 1)
                img = Preprocessor.homomorph(img, 16, 1, gamma2)

        totensor = T.ToTensor()
        img = totensor(img)
        gt = totensor(gt)

        if (self.is_train):
            img = Preprocessor.T_img(img)

        return {'name': name, 'img': img, 'gt': gt}
    

# main function
# augmentation transform plot 이미지 파일로 저장
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SegmentationDataset(
        img_dir='data/train/img',
        gt_dir='data/train/gt',
        is_train=True
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False
    )

    n_batch = len(dataloader)

    for i, batch in tqdm(enumerate(dataloader)):
        img = batch['img'].to(device=device, dtype=torch.float32)
        gt = batch['gt'].to(device=device, dtype=torch.long)

        visualizer.visualize(
            suptitle=f'Image Augmentation({i + 1} of {n_batch})',
            displays=[img, gt.squeeze(dim=1).to(torch.float)],
            loss=None,
            score_acc=None,
            score_miou=None,
            score_fwiou=None,
            column_titles=['img', 'gt'],
            output_dir='data/output/aug',
            output_name=f'aug_{i + 1}'
        )