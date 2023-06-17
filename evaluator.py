import torch
from torch import nn
from torch.utils.data import DataLoader

import cv2
import numpy as np
import os
from tqdm import tqdm

from metrics import Accuracy, MIoU, FwIoU
from dataset import SegmentationDataset
import net_selector
import visualizer


def evaluate(
    device,
    net,
    net_name,
    checkpoint_name,
    dataset,
    n_class,
    suptitle,
    output_dir,
    output_name
):
    checkpoint_filedir = f'checkpoints/{net_name}/{checkpoint_name}.pt'

    if not os.path.exists(checkpoint_filedir):
        print('Error: Checkpoint file does not exist.')

        return

    checkpoint = torch.load(checkpoint_filedir)

    net.load_state_dict(checkpoint['net_state_dict'])

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False
    )

    if not os.path.exists(f'{output_dir}/visualization'): os.makedirs(f'{output_dir}/visualization')

    if not os.path.exists(f'{output_dir}/pred/{output_name}'): os.makedirs(f'{output_dir}/pred/{output_name}')

    criterion = nn.CrossEntropyLoss()
    metric_acc = Accuracy(n_class=n_class)
    metric_miou = MIoU(n_class=n_class)
    metric_fwiou = FwIoU(n_class=n_class)

    loss_sum = 0
    score_acc_sum = 0
    score_miou_sum = 0
    score_fwiou_sum = 0
    n_batch = len(dataloader)

    print(suptitle)

    net.eval()

    for i, batch in tqdm(enumerate(dataloader)):
        name = batch['name'][0]
        img = batch['img'].to(device=device, dtype=torch.float32)
        gt = batch['gt'].to(device=device, dtype=torch.long)

        logit_pred = net(img)

        chunk_gt = torch.chunk(gt, n_class, dim=1)
        maxval_gt, _ = torch.max(torch.cat(chunk_gt, dim=1), dim=1, keepdim=True)

        class_gt = torch.zeros_like(maxval_gt)

        for j in range(n_class): class_gt[maxval_gt == chunk_gt[j]] = j

        class_gt = class_gt.squeeze(dim=1)

        loss = criterion(logit_pred, class_gt)
        loss_sum += loss.item()

        # metrics
        class_pred = torch.argmax(logit_pred, dim=1).to(torch.float)
        score_acc = metric_acc(class_pred, class_gt)
        score_acc_sum += score_acc.item()
        score_miou = metric_miou(class_pred, class_gt)
        score_miou_sum += score_miou.item()
        score_fwiou = metric_fwiou(class_pred, class_gt)
        score_fwiou_sum += score_fwiou.item()

        pred = torch.zeros_like(gt)

        for j in range(n_class): pred[:, j] = (class_pred == j).type(torch.uint8)

        pred_mask = pred[0].to(torch.float).detach().cpu().numpy()
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        pred_mask *= 255

        cv2.imwrite(f'{output_dir}/pred/{output_name}/{name}.PNG', pred_mask)

        visualizer.visualize(
            suptitle=f'{suptitle}({name}, {i + 1} of {n_batch})',
            displays=[img, pred.to(torch.float), gt.squeeze(dim=1).to(torch.float)],
            loss=loss,
            score_acc=score_acc,
            score_miou=score_miou,
            score_fwiou=score_fwiou,
            column_titles=['img', 'pred', 'gt'],
            output_dir=f'{output_dir}/visualization',
            output_name=f'{output_name}_{i + 1}'
        )

    print(f'loss:     {loss_sum / len(dataloader): .4f}')
    print(f'accuracy: {score_acc_sum / len(dataloader): .4f}')
    print(f'mIoU:     {score_miou_sum / len(dataloader): .4f}')
    print(f'fwIoU:    {score_fwiou_sum / len(dataloader): .4f}')
    print('----')


# main function
# evaluation score 및 inference plot 이미지 파일로 저장
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_name = 'attention_unet'
    n_class = 3
    c_x1 = 32
    padding_mode = 'zeros'

    net = net_selector.nets_dict[net_name].Net(n_class, c_x1, padding_mode).to(device)

    train_dataset = SegmentationDataset(
        img_dir='data/train/img',
        gt_dir='data/train/gt',
        is_train=True
    )

    test_dataset = SegmentationDataset(
        img_dir='data/test/img',
        gt_dir='data/test/gt',
        is_train=False
    )

    print('----')
    
    for checkpoint in os.listdir(f'checkpoints/{net_name}'):
        checkpoint_name = checkpoint[:-3]

        print(f'Evaluation of \'{checkpoint}\'')
        print('----')


        evaluate(
            device=device,
            net=net,
            net_name=net_name,
            checkpoint_name=checkpoint_name,
            dataset=train_dataset,
            n_class=n_class,
            suptitle='Inference at Training Dataset',
            output_dir=f'data/output/{checkpoint_name}',
            output_name='train'
        )

        evaluate(
            device=device,
            net=net,
            checkpoint_name=checkpoint_name,
            net_name=net_name,
            dataset=test_dataset,
            n_class=n_class,
            suptitle='Inference at Test Dataset',
            output_dir=f'data/output/{checkpoint_name}',
            output_name='test'
        )