"""
utility.py
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def imshow(img: np.ndarray, caption: str, grid=True, grid_size: int = 64) -> None:
    """
    Jupyter用の画像表示関数。
    cv.imshow()はJupyter上では動作しないため、その代替関数。

    Args:
        img : 画像
        caption (str): 画像のキャプション
        grid (bool, optional): グリッド線を表示するかどうか. Defaults to True.
        grid_size (int, optional): グリッド線の間隔. Defaults to 64.
    """
    plt.figure(figsize=(6, 6))
    plt.tick_params(labelsize=7)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.xticks(np.arange(0, img.shape[1], grid_size))
    plt.yticks(np.arange(0, img.shape[0], grid_size))
    if grid:
        plt.grid(which="both", color="#00FF00", linestyle="-", alpha=0.7)  # グリッド表示

    print(f"{caption} - shape:", img.shape)
    plt.show()