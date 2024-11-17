"""
file_utility.py
"""

import os
import shutil
from glob import glob

def enum_files(path: str, pattern: str = "*", recursive=False):
    """
    フォルダ内のファイル一覧を昇順にソートして取得する。
    patternを指定すると、パターンに一致するファイルのみ取得する。
    
    Hint:
        pathに含まれるフォルダ配下のファイルを検索したい場合（それ以上の深い階層は対象外で良い場合）は
        recursive=Falseの状態で、patternに"*/*.png"のように指定すれば、サブフォルダまでは検索可能。
    
    Args:
        path (str): 検索対象のフォルダ
        pattren (str, optional): ファイル名のパターン. Defaults to None.
        recursive (bool, optional): 配下全てのフォルダも検索するかどうか. Default to False.
    
        Returns:
            list: ファイル名のリスト
    """

    files = glob(os.path.join(path, pattern), recursive=recursive)
    files.sort()
    return files


def move_files_to_folder(list_of_files: list, dst_dir: str, mode: str = "train") -> None:
    """
    フォルダ内のファイルを指定フォルダに移動する。

    Args:
        list_of_files (list): 移動対象のファイル名リスト
        dst_dir (str): 移動先のフォルダ
    """
    # 指定フォルダに移動
    for f in list_of_files:
        try:
            if mode == "train":
                shutil.copy(f, dst_dir)
            elif mode == "val":
                new_filename = os.path.join(dst_dir, os.path.basename(f).replace("train", "val"))
                shutil.copy(f, new_filename)
            
        except:
            print(f)
            assert False