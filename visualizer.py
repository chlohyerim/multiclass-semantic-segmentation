import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize(
    suptitle,
    displays,
    column_titles,
    output_dir,
    output_name
):
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    n_row = len(displays[0])
    n_column = len(column_titles)
    fig = plt.figure(figsize=(3 * n_column, 3 * n_row))

    fig.suptitle(suptitle)

    for i in range(n_row):
        for j in range(n_column):
            ax = fig.add_subplot(n_row, n_column, i * n_column + j + 1)

            if i == 0: ax.set_title(column_titles[j])

            display_np = displays[j][i].detach().cpu().numpy()
            display_np = np.transpose(display_np, (1, 2, 0))
            display_np = cv2.cvtColor(display_np, cv2.COLOR_BGR2RGB)

            ax.imshow(display_np)
            ax.axis('off')
        
    fig.tight_layout()
    plt.close()

    fig.savefig(f'{output_dir}/{output_name}.PNG')