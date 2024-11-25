# 画像に写ったダイスを検出し、出目合計を出力する物体認識モデル

## 概要
  * 画像に写ったダイス（1～3個）をYOLOv8mで検出し、出目合計を算出する。
  * labelimgツールを使ってアノテーション作業を実施。
  * 開催終了間際（締切り2Week前）に参加し、コンペ順位は全体の中央付近。

## 取組みのポイント
  * 今回のコンペ内容は物体認識が優位と考えて、YOLOを採用。
  * 訓練データが200,000枚（ダイス1～2個）あり、EDAでピックアップする画像を600枚選定。
    ※アノテーション作業：1～2日（約5時間）で作業可能な枚数を見積もり、600枚に設定。
  * 1～6の出目を均等に学習させるため、ダイス2個の画像かつ出目合計が2～12となる画像を以下数量でピックアップ。
    "2: 19枚、3: 32枚、4: 62枚、5: 66枚、6: 79枚、7: 84枚、8: 79枚、9: 66枚、10: 62枚、11: 32枚、12: 19枚"
    ※出目組み合わせの確率を計算し、1～6の出現期待値が各200個になるよう画像枚数を決定。
  * 最終局面では学習済みモデルに訓練データを入力し、アノテーションデータを自動生成することで、学習データを2,400枚まで拡張。
  * テストデータにノイズが付与されており、AutoEncorderによるノイズ除去を試行。画像がボヤけて出目を誤る可能性があり採用を見送った。
    
## 解法 
### 前処理
  * 訓練データ画像から、200,000枚のダイス個数1個、2個を分別（白ドット数のヒストグラムを取り、閾値を設定）
  * ダイス2個の画像に絞り、出目合計2～12をサンプリングでピックアップ
  * labelimgでラベリング作業の実施

### モデル学習
  * YOLOv8mを採用
  * テストデータのノイズを再現するため、訓練データにもランダムでガウシアンノイズを付与
  　※AutoEncoderのノイズ除去に失敗したため、ノイズも含めて学習させる方法を選択
  * Mixup/mosaicを使用し、最後の10epochを無効可して学習

### 推論
  * テストデータを入力し、ダイスの出目結果を収集
  
### 後処理
  * ダイスの出目を合計し、推論結果として保存

## 使用ライブラリ
  * numpy
  * pandas
  * matplotlib
  * sklearn
  * datetime
  * seaborn
  * OpeCV
  * PIL
  * tqdm
  * torch
  * torchvision
  * tensorboard
  * ultralytics
