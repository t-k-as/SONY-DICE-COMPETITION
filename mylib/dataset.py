"""
dataset.py
"""

import os

from .annotation import find_annotation_file
from .file_utility import enum_files

# クラスラベル
CLASSES = ["1", "2", "3", "4", "5", "6"]
TRAIN_NUMBER = 200000
TEST_NUMBER = 24922

def make_yolo_files(output_dir: str, mode: str="train"):
    """
    YOLO系の学習などで利用する各種ファイルを作成する。
    - classes.txt
    - train.txt
    - val.txt
    - test.txt
    - dataset.yaml

    Args:
        outout_dir (str): 出力先ディレクトリ
    """

    # classes.txtを作成
    # クラスラベルを定義したファイル
    if mode == "train":
        with open(os.path.join(output_dir, "classes.txt"), "w") as f:
            f.write("\n".join(CLASSES))

        # train.txtを作成
        # 学習に利用する0～200,000のjpgファイル(一部)が対象
        with open(os.path.join(output_dir, "train.txt"), "w") as f:
            #for num in range(TRAIN_NUMBER):
            #    img_files = enum_files(output_dir, f"*/train_image{num}.png")
            #    if img_files:
            #        img_files[0] = "./images/" + f"train_image{num}.png"
            #        #f.write("\n".join(os.path.basename(img_files)) + "\n")
            #        f.write("\n".join(img_files) + "\n")
            img_files = enum_files(output_dir, "*/train_image*.png")
            img_files = [file.replace("/data/yolo_640", "") for file in img_files]
            f.write("\n".join(img_files) + "\n")

        # val.txtを作成
        # train用のデータをval用として使用する
        with open(os.path.join(output_dir, "val.txt"), "w") as f:
            #for num in range(TRAIN_NUMBER):
            #    img_files = enum_files(output_dir, f"*/val_image{num}.png")
            #    if img_files:
            #        img_files[0] = "./images/" + f"val_image{num}.png"
            #        #f.write("\n".join(os.path.basename(img_files)) + "\n")
            #        f.write("\n".join(img_files) + "\n")
            img_files = enum_files(output_dir, "*/val_image*.png")
            img_files = [file.replace("/data/yolo_640", "") for file in img_files]
            f.write("\n".join(img_files) + "\n")
                
        # dataset.yamlを作成
        # 学習時に利用するデータセットの情報を定義したファイル
        dataset_yaml = f"""
        path: {output_dir}
        train: train.txt
        val: val.txt
        nc: {len(CLASSES)}
        names: {CLASSES}
        """

        with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
            f.write(dataset_yaml)

    elif mode == "test":            
        # test.txtを作成
        # 提出対象となるtest用の画像ファイルが対象
        with open(os.path.join(output_dir, "test.txt"), "w") as f:
            for num in range(TEST_NUMBER):
                img_files = enum_files(output_dir, f"*/test_image{num}.png")
                img_files[0] = "./images/" + f"test_image{num}.png"
                #f.write("\n".join(os.path.basename(img_files)) + "\n")
                f.write("\n".join(img_files) + "\n")


def get_dataset(img_dir: str):
    """
    指定ディレクトリの画像と、アノテーションファイルを取得する。

    Args:
        img_dir (str): 画像ディレクトリ
    
    Returns:
        list[tuple[str, str]]: img_file, anno_fileのタプルのリスト
    """

    items = []
    for img_file in enum_files(img_dir, "*.png"):
        ano_file = find_annotation_file(img_file)

        # img, annotationをタプルにセットする
        items.append(((img_file, ano_file)))

    
    return items