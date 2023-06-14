import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def visualize(
    suptitle,
    displays,
    loss,
    score_acc,
    score_miou,
    score_fwiou,
    column_titles,
    output_dir,
    output_name
):
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    n_row = len(displays[0])
    n_column = len(column_titles)
    fig = plt.figure(figsize=(4 * n_column, 5 * n_row))

    fig.suptitle(suptitle)

    for i in range(n_row):
        for j in range(n_column):
            ax = fig.add_subplot(n_row, n_column, i * n_column + j + 1)

            if i == 0: ax.set_title(column_titles[j])

            display_np = displays[j][i].detach().cpu().numpy()
            display_np = np.transpose(display_np, (1, 2, 0))
            display_np = cv2.cvtColor(display_np, cv2.COLOR_BGR2RGB)

            if j == 2:
                textstr = '\n'.join((
                    f'CE loss={loss: .4f}',
                    f'Accuracy={score_acc: .4f}',
                    f'mIoU={score_miou: .4f}',
                    f'fwIoU={score_fwiou: .4f}'
                ))
                # textstr = f'CE loss={loss: .4f}\nfwIoU={score_fwiou: .4f}'
                bbox_props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                ax.text(0.6, 1.4, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=bbox_props)

            ax.imshow(display_np)
            ax.axis('off')
        
    fig.tight_layout()
    plt.close()

    fig.savefig(f'{output_dir}/{output_name}.PNG')