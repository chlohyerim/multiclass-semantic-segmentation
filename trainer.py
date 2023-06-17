import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import os
from tqdm import tqdm

from dataset import SegmentationDataset
from metrics import MIoU, FwIoU
import net_selector


def train(
    device,
    net,
    net_name,
    train_dataset,
    validation_dataset,
    n_class,
    padding_mode,
    n_epoch,
    batch_size,
    learning_rate
):
    now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # now = '2023_06_14_23_55_49'
    checkpoint_name = f'{net_name}_{padding_mode}_{now}'
    writer = SummaryWriter(f'runs/{checkpoint_name}')
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    optimizer = optim.AdamW(
        params=net.parameters(),
        lr=learning_rate
    )

    criterion = nn.CrossEntropyLoss()
    # metric_miou = MIoU(n_class=n_class)
    metric_fwiou = FwIoU(n_class=n_class)

    epoch_init = 0
    validation_score_max = 0

    # initializing checkpoint
    checkpoints_dir = f'checkpoints/{net_name}'

    if not os.path.exists(checkpoints_dir): os.makedirs(checkpoints_dir)

    checkpoint_filedir = f'{checkpoints_dir}/{checkpoint_name}.pt'

    if os.path.exists(checkpoint_filedir):
        checkpoint = torch.load(checkpoint_filedir)

        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch_init = checkpoint['epoch']
        validation_score_max = checkpoint['validation_score_max']

    for epoch in tqdm(range(epoch_init + 1, n_epoch + 1)):        
        # training
        net.train()

        loss_sum = 0
        # score_miou_sum = 0
        score_fwiou_sum = 0

        for _, batch in enumerate(train_dataloader):
            img = batch['img'].to(device=device, dtype=torch.float32)
            gt = batch['gt'].to(device=device, dtype=torch.long)

            logit_pred = net(img)

            chunk_gt = torch.chunk(gt, n_class, dim=1)
            maxval_gt, _ = torch.max(torch.cat(chunk_gt, dim=1), dim=1, keepdim=True)

            class_gt = torch.zeros_like(maxval_gt)

            for i in range(n_class): class_gt[maxval_gt == chunk_gt[i]] = i

            class_gt = class_gt.squeeze(dim=1)
            
            loss = criterion(logit_pred, class_gt)
            loss_sum += loss.item()

            class_pred = torch.argmax(logit_pred, dim=1).to(torch.float)
            # score_miou_sum += metric_miou(class_pred, class_gt).item()
            score_fwiou_sum += metric_fwiou(class_pred, class_gt).item()
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_loss = loss_sum / len(train_dataloader)
        # train_score = score_miou_sum / len(train_dataloader)
        train_score = score_fwiou_sum / len(train_dataloader)

        writer.add_scalar('training loss/epoch', train_loss, epoch)
        writer.add_scalar('training score/epoch', train_score, epoch)

        # validation with part of test data
        net.eval()

        with torch.no_grad():
            loss_sum = 0
            # score_miou_sum = 0
            score_fwiou_sum = 0

            for _, batch in enumerate(validation_dataloader):
                img = batch['img'].to(device=device, dtype=torch.float32)
                gt = batch['gt'].to(device=device, dtype=torch.long)

                logit_pred = net(img)

                chunk_gt = torch.chunk(gt, n_class, dim=1)
                maxval_gt, _ = torch.max(torch.cat(chunk_gt, dim=1), dim=1, keepdim=True)

                class_gt = torch.zeros_like(maxval_gt)

                for i in range(n_class): class_gt[maxval_gt == chunk_gt[i]] = i

                class_gt = class_gt.squeeze(dim=1)

                loss = criterion(logit_pred, class_gt)
                loss_sum += loss.item()
                
                class_pred = torch.argmax(logit_pred, dim=1).to(torch.float)
                # score_miou_sum += metric_miou(class_pred, class_gt).item()
                score_fwiou_sum += metric_fwiou(class_pred, class_gt).item()

            validation_loss = loss_sum / len(validation_dataloader)
            # validation_score = score_miou_sum / len(validation_dataloader)
            validation_score = score_fwiou_sum / len(validation_dataloader)

            # update model to save if maximum test score is updated
            if validation_score > validation_score_max:
                validation_score_max = validation_score

                writer.add_scalar('validation loss/epoch', validation_loss, epoch)
                writer.add_scalar('validation score/epoch', validation_score, epoch)

                print('')
                print(f'Saved model at epoch {epoch}.')
                print(f'training loss:    {train_loss: .4f}')
                print(f'training score:   {train_score: .4f}')
                print(f'validation loss:  {validation_loss: .4f}')
                print(f'validation score: {validation_score: .4f}')

                torch.save(
                    {'net_state_dict': net.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'epoch': epoch,
                     'validation_score_max': validation_score_max},
                    checkpoint_filedir
                )


# main function
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_name = 'attention_unet'
    n_class = 3
    c_x1 = 32
    padding_mode = 'replicate'

    net = net_selector.nets_dict[net_name].Net(n_class, c_x1, padding_mode).to(device)

    train_dataset = SegmentationDataset(
        img_dir='data/train/img',
        gt_dir='data/train/gt',
        is_train=True
    )

    validation_dataset = SegmentationDataset(
        img_dir='data/validation/img',
        gt_dir='data/validation/gt',
        is_train=False
    )

    train(
        device=device,
        net=net,
        net_name=net_name,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        n_class=n_class,
        padding_mode=padding_mode,
        n_epoch=2000,
        batch_size=8,
        learning_rate=1e-4
    )