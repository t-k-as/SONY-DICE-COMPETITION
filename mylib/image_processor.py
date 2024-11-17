"""
image_processor.py
"""

import cv2
import numpy as np

from .utility import imshow

def make_resize(img: np.ndarray, resize: int=512, verbose: bool=False):
    """
    画像をリサイズする。
    YOLO形式のアノテーション座標変換に必要なシフト量と縮尺比率も返す。
    
    Args:
        img (np.ndarray): 画像
        expand_ratio (float, optional): 基盤領域の拡張率. Defaults to 1.10.
        resize (int, optional): リサイズ後のサイズ. Defaults to 512.
        verbose (bool, optional): 詳細な処理情報を表示するかどうか. Defaults to False.
    
    Returns:
        (resize_img, ratio_info): リサイズ画像と座標変換に必要なシフト亮と縮尺比率
            resize_img: 処理後の画像
            ratio_info: 座標変換に必要なシフト量と縮尺比率の辞書型（ratio_w, ratio_h）
    """
    if verbose:
        imshow(img, "元画像", grid_size=300)

     # リサイズ
    resize_img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_NEAREST)
    if verbose:
        imshow(resize_img, "リサイズ後の画像")

    return resize_img