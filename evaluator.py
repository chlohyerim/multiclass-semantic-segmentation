import torch
from torch import nn
from torch.utils.data import DataLoader

import os
from tqdm import tqdm

from metrics.miou_score import MIoUScore
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
    batch_size,
    suptitle,
    output_dir,
    output_name
):
    checkpoint_filedir = f'checkpoints/{net_name}/{checkpoint_name}.pt'

    if not os.path.exists(checkpoint_filedir):
        print('Error: Checkpoint file does not exist.')

        return

    checkpoint = torch.load(checkpoint_filedir)

    net.load_state_dict(checkpoint['state_dict'])

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False
    )

    if not os.path.exists(output_dir): os.makedirs(output_dir)

    criterion = nn.CrossEntropyLoss()
    metric_miou = MIoUScore(n_class=n_class)

    running_loss = 0
    score_miou = 0
    n_batch = len(dataloader)

    net.eval()

    for i, batch in tqdm(enumerate(dataloader)):
        img = batch['img'].to(device=device, dtype=torch.float32)
        gt = batch['gt'].to(device=device, dtype=torch.long)

        logit_pred = net(img)

        chunk_gt = torch.chunk(gt, n_class, dim=1)
        maxval_gt, _ = torch.max(torch.cat(chunk_gt, dim=1), dim=1, keepdim=True)

        class_gt = torch.zeros_like(maxval_gt)

        for j in range(n_class): class_gt[maxval_gt == chunk_gt[j]] = j

        loss = criterion(logit_pred, class_gt.squeeze(dim=1))
        running_loss += loss.item()

        class_pred = torch.argmax(logit_pred, dim=1).to(torch.float)
        score_miou += metric_miou(class_pred, class_gt.squeeze(dim=1)).item()

        pred = torch.zeros_like(gt)

        for j in range(n_class): pred[:, j] = (class_pred == j).type(torch.uint8)

        visualizer.visualize(
            suptitle=f'{suptitle}({i + 1} of {n_batch})',
            displays=[img, pred.to(torch.float), gt.squeeze(dim=1).to(torch.float)],
            column_titles=['img', 'pred', 'gt'],
            output_dir=output_dir,
            output_name=f'{output_name}_{i + 1}'
        )

    print(f'loss of \'{suptitle}\' : {running_loss / len(dataloader): .4f}')
    print(f'score of \'{suptitle}\': {score_miou / len(dataloader): .4f}')
    print('-----------------------------------------------')


# main function
# evaluation score 및 inference plot 이미지 파일로 저장
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_name = 'attention_unet'
    n_class = 3
    cout_encoder1 = 32

    net = net_selector.nets_dict[net_name].Net(n_class, cout_encoder1).to(device)

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
    
    for checkpoint in os.listdir(f'checkpoints/{net_name}'):
        checkpoint_name = checkpoint[:-3]

        evaluate(
            device=device,
            net=net,
            net_name=net_name,
            checkpoint_name=checkpoint_name,
            dataset=train_dataset,
            n_class=3,
            batch_size=4,
            suptitle='Training Visualization',
            output_dir=f'data/output/{checkpoint_name}',
            output_name='train'
        )

        evaluate(
            device=device,
            net=net,
            checkpoint_name=checkpoint_name,
            net_name=net_name,
            dataset=test_dataset,
            n_class=3,
            batch_size=4,
            suptitle='Inference at Test Data',
            output_dir=f'data/output/{checkpoint_name}',
            output_name='test'
        )