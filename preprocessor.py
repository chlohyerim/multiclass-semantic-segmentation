import torch
import torch.nn.functional as F
import torchvision.transforms as T

import cv2
import numpy as np


class Preprocessor:
    def interpolateGt(src):
        b_src, g_src, r_src = cv2.split(src)
        maxval_src = np.maximum(np.maximum(r_src, g_src), b_src)

        dst = np.zeros_like(src)
        dst[maxval_src == b_src] = [255, 0, 0]
        dst[maxval_src == g_src] = [0, 255, 0]
        dst[maxval_src == r_src] = [0, 0, 255]

        return dst


    def flipHorizontal(src): return cv2.flip(src, 1)


    def perspective(src, pts_src, is_gt):
        h, w = src.shape[:2]
        pts_dst = np.float32(
            [[0, 0],
             [w - 1, 0],
             [w - 1, h - 1],
             [0, h - 1]]
        )
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)

        dst = cv2.warpPerspective(src, M, (w, h))

        return Preprocessor.interpolateGt(dst) if is_gt else dst


    def resizedcrop(src, scale, w_scaled, h_scaled, x, y, is_gt):
        h, w = src.shape[:2]

        src = cv2.resize(src, dsize=(w_scaled, h_scaled), interpolation=cv2.INTER_AREA)

        if scale > 1:
            dst = src[y:(y + h), x:(x + w)]
        elif scale < 1:
            border_y = (h - h_scaled) // 2
            border_x = (w - w_scaled) // 2

            dst = cv2.copyMakeBorder(src, top=border_y, bottom=border_y, left=border_x, right=border_x, borderType=cv2.BORDER_REPLICATE)

        return Preprocessor.interpolateGt(dst) if is_gt else dst
    

    def rotate(src, angle, w_scaled, h_scaled, is_gt):
        h, w = src.shape[:2]
        y = (h_scaled - h) // 2
        x = (w_scaled - w) // 2

        dst = cv2.copyMakeBorder(src, top=y, bottom=y, left=x, right=x, borderType=cv2.BORDER_REFLECT)

        M = cv2.getRotationMatrix2D(center=(int(w_scaled / 2), int(h_scaled / 2)), angle=angle, scale=1)

        dst = cv2.warpAffine(dst, M, (w_scaled, h_scaled))
        dst = dst[y:(y + h), x:(x + w)]
        
        return Preprocessor.interpolateGt(dst) if is_gt else dst
    

    def equalizeHist(src):
        dst_yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
        dst_yuv[:, :, 0] = cv2.equalizeHist(dst_yuv[:, :, 0])

        return cv2.cvtColor(dst_yuv, cv2.COLOR_YUV2BGR)
    

    def homomorph(src, sigma, gamma1, gamma2):
        src_yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
        y_src = src_yuv[:, :, 0]
        ylog_src = np.log1p(np.array(y_src) / 255)

        h, w = y_src.shape[:2]
        M = 2 * h + 1
        N = 2 * w + 1

        (Y, X) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
        Yc = np.ceil(N / 2)
        Xc = np.ceil(M / 2)
        gaussianNumerator = (X - Xc) ** 2 + (Y - Yc) ** 2

        lpf = np.exp(-gaussianNumerator / (2 * sigma * sigma))
        hpf = 1 - lpf

        lpfshift = np.fft.ifftshift(lpf.copy())
        hpfshift = np.fft.ifftshift(hpf.copy())

        yfft_src = np.fft.fft2(ylog_src.copy(), (M, N))
        ylf_src = np.real( np.fft.ifft2( yfft_src.copy() * lpfshift, (M, N) ) )
        yhf_src = np.real( np.fft.ifft2( yfft_src.copy() * hpfshift, (M, N) ) )

        y_dst = gamma1 * ylf_src[0:h, 0:w] + gamma2 * yhf_src[0:h, 0:w]
        
        y_dst = np.expm1(y_dst)
        y_dst = (y_dst - np.min(y_dst)) / (np.max(y_dst) - np.min(y_dst))
        y_dst = np.array(255 * y_dst, dtype='uint8')

        src_yuv[:, :, 0] = y_dst

        dst = cv2.cvtColor(src_yuv, cv2.COLOR_YUV2BGR)
        
        return dst


    T_img = T.Compose([
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.25),
    ])
