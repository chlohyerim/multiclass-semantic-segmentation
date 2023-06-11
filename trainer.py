import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import os
from tqdm import tqdm

from dataset import SegmentationDataset
from metrics.miou_score import MIoUScore
import net_selector


def train(
    device,
    net,
    net_name,
    train_dataset,
    validation_dataset,
    n_class,
    n_epoch,
    batch_size,
    learning_rate
):
    now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    checkpoint_name = f'{net_name}_{now}'
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
    metric_miou = MIoUScore(n_class=n_class)

    validation_score_max = 0

    checkpoints_dir = f'checkpoints/{net_name}'

    if not os.path.exists(checkpoints_dir): os.makedirs(checkpoints_dir)

    checkpoint_filedir = f'{checkpoints_dir}/{checkpoint_name}.pt'

    for epoch in tqdm(range(1, n_epoch + 1)):        
        # training
        net.train()

        running_loss = 0
        score_miou = 0

        for _, batch in enumerate(train_dataloader):
            img = batch['img'].to(device=device, dtype=torch.float32)
            gt = batch['gt'].to(device=device, dtype=torch.long)

            logit_pred = net(img)

            chunk_gt = torch.chunk(gt, n_class, dim=1)
            maxval_gt, _ = torch.max(torch.cat(chunk_gt, dim=1), dim=1, keepdim=True)

            class_gt = torch.zeros_like(maxval_gt)

            for i in range(n_class): class_gt[maxval_gt == chunk_gt[i]] = i
            
            loss = criterion(logit_pred, class_gt.squeeze(dim=1))
            running_loss += loss.item()

            class_pred = torch.argmax(logit_pred, dim=1).to(torch.float)
            score_miou += metric_miou(class_pred, class_gt.squeeze(1)).item()
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_loss = running_loss / len(train_dataloader)
        train_score = score_miou / len(train_dataloader)

        writer.add_scalar('training loss/epoch', train_loss, epoch)
        writer.add_scalar('training score/epoch', train_score, epoch)

        # validation with part of test data
        net.eval()

        with torch.no_grad():
            running_loss = 0
            score_miou = 0

            for _, batch in enumerate(validation_dataloader):
                img = batch['img'].to(device=device, dtype=torch.float32)
                gt = batch['gt'].to(device=device, dtype=torch.long)

                logit_pred = net(img)

                chunk_gt = torch.chunk(gt, n_class, dim=1)
                maxval_gt, _ = torch.max(torch.cat(chunk_gt, dim=1), dim=1, keepdim=True)

                class_gt = torch.zeros_like(maxval_gt)

                for i in range(n_class): class_gt[maxval_gt == chunk_gt[i]] = i

                loss = criterion(logit_pred, class_gt.squeeze(dim=1))
                running_loss += loss.item()
                
                class_pred = torch.argmax(logit_pred, dim=1).to(torch.float)
                score_miou += metric_miou(class_pred, class_gt.squeeze(1)).item()

            validation_loss = running_loss / len(validation_dataloader)
            validation_score = score_miou / len(validation_dataloader)

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
                    {'epoch': epoch,
                     'state_dict': net.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'train_loss': train_loss,
                     'train_score': train_score,
                     'validation_loss': validation_loss,
                     'validation_score': validation_score},
                    checkpoint_filedir
                )


# main function
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_name = 'unet'
    n_class = 3
    cout_encoder1 = 32

    net = net_selector.nets_dict[net_name].Net(n_class, cout_encoder1).to(device)

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
        n_class=3,
        n_epoch=2000,
        batch_size=8,
        learning_rate=1e-4
    )