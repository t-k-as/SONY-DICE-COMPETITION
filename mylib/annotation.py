"""
annotation.py
"""

import os

import cv2
import numpy as np
import pandas as pd

from .file_utility import enum_files
from .utility import imshow


def find_annotation_file(img_file: str):
    """
    画像ファイルのパスからアノテーションファイルを検索して返す。
    アノテーションファイルは、../labelsフォルダから探す。

    Args:
        img_file (str): 画像ファイル名
    
    Returns:
        str: アノテーションファイル（見つからない場合はNone）
    """

    img_dir = os.path.dirname(img_file) # ex: /data_folder/images
    anno_name = os.path.basename(img_file).replace(".jpg", ".txt").replace(".png", ".txt") # ex: train_image0.txt

    # 検索するアノテーションファイル名のリスト作成
    anno_files = [
        os.path.join(img_dir, "../labels", anno_name),
        os.path.join(img_dir, anno_name),
    ]

    # アノテーションファイル名を検索して存在すれば返す
    for anno_file in anno_files:
        if os.path.exists(anno_file) and anno_file.endswith(".txt"):
            return anno_file
    return None # Not found


def load_predict(anno_file: str, with_filename=True):
    """
    YOLO形式の推論結果ファイルを読み込んでDataFrame形式で取得する。

    Args:
        anno_file (str): 推論結果ファイル名（YOLO形式）
        with_filename (bool, optional): ファイル名を含めるかどうか. Defaults to True.
        
    Returns:
        pd.DataFrame: 推論結果のDataFrame（YOLO形式）
            項目は["class", "x", "y", "w", "h", "conf"]が含まれる（YOLOの比率形式）
            ファイル名を含める場合は"filename"も追加される
    """

    # 推論結果の読み込み
    df = pd.read_csv(anno_file, sep=" ", header=None, names=["class", "x", "y", "w", "h", "conf"])
    if with_filename:
        df.insert(0, "filename", os.path.basename(anno_file)) # 最初の列
    return df


def load_predicts(anno_dir: str, with_filename=True):
    """
    指定ディレクトリのYOLO形式の推論結果ファイルを読み込んでDataFrame形式で取得する。

    Args:
        anno_dir (str): 推論結果ファイルのディレクトリ（YOLO形式）
        with_filename (bool, optional): ファイル名を含めるかどうか. Defaults to True.
    
    Returns:
        pd.DataFrame: 推論結果のDetaFrame（YOLO形式）
            項目は["class", "x", "y", "w", "h", "conf"]が含まれる（YOLOの比率形式）
            ファイル名を含める場合は"filename"も追加される
    """
    # 推論結果を読み込んで統合する
    df = pd.DataFrame()
    for anno_file in enum_files(anno_dir, "*.txt"):
        df = pd.concat([df, load_predict(anno_file, with_filename)])
    return df


def load_annotation(anno_file: str, ratio_info: dict = None):
    """
    YOLO形式のアノテーションファイルを読み込んでDetaFrame形式で取得する。
    ratio_infoが指定されている場合は、座標変換も行う。

    Args:
        anno_file (str): アノテーションファイル名
        ratio_info (dict, optional): make_resize()で取得した座標変換情報. Defaults to None.
    
    Returns:
        pd.DataFrame: アノテーション情報（YOLO形式）
            項目は["class", "x", "y", "w", "h"]が含まれる（YOLOの比率形式）
    """
    # YOLO形式のアノテーションファイルを読み込む
    df = pd.read_csv(anno_file, sep=" ", header=None, names=["class", "x", "y", "w", "h"])
    
    return df


def save_annotation(anno_file: str, df_annotation: pd.DataFrame, override: bool=False):
    """
    DataFrame形式のアノテーション情報をYOLO形式のアノテーションファイルに保存する。

    Args;
        anno_file (str): 保存先のアノテーションファイル名
        df_annotation (pd.DataFrame): アノテーション情報（YOLO形式）
            項目は["class", "x", "y", "w", "h"]を含んでいること
        override (bool, optional): 既存のファイルを上書きするかどうか. Defaults to False.
    """
    # YOLO形式のアノテーションファイルを保存
    if os.path.exists(anno_file) and not override:
        raise Exception(f"Annotation file already exists: {anno_file}")
    df_annotation.to_csv(anno_file, sep=" ", header=False, index=False)


def yolo_to_real(img: np.ndarray, df_annotation: pd.DataFrame):
    """
    YOLO形式のアノテーション情報を実座標に変換する。

    Args:
        img (np.ndarray): 画像
        df_annotation (pd.DataFrame): アノテーション情報（YOLO形式）
    
    Returns:
        pd.DataFrame: アノテーション情報（実座標）
    """
    # アノテーション座標（比率）を実座標に変換（四捨五入して整数にする）
    h, w = img.shape[:2]
    df = df_annotation.copy()
    df["x"] = np.round(df["x"] * w).astype(int)
    df["y"] = np.round(df["y"] * h).astype(int)
    df["w"] = np.round(df["w"] * w).astype(int)
    df["h"] = np.round(df["h"] * h).astype(int)
    return df


def show_annotation(img: np.ndarray, df_annotation: pd.DataFrame):
    """
    画像にアノテーション情報を描画する。

    Args:
        img (np.ndarray): 画像
        df_annotation (pd.DataFrame): アノテーション情報（YOLO形式）
    """
    # アノテーション座標（比率）を実座標に変換
    df = yolo_to_real(img, df_annotation)

    # 描画色（6クラス前提）
    colors = [(0, 255, 0), (255, 64, 255), (0, 120, 255), (64, 64, 255), (0, 0, 255), (255, 0, 0)]

    # アノテーション情報の描画
    img_preview = img.copy()
    bboxes = df[["class", "x", "y", "w", "h"]].values.astype(int)
    for bbox in bboxes:
        clazz, x, y, w, h = bbox
        cv2.rectangle(
            img_preview,
            (int(x - w / 2), int(y - h / 2)),
            (int(x + w / 2), int(y + h / 2)),
            colors[clazz],
            2,
        )
    imshow(img_preview, "Annotation", grid=False)

