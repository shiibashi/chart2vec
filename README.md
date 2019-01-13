## インストール
```
pip install mpl-finance
pip install git+https://github.com/shiibashi/myautokeras.git
pip install pillow

```
## python
3.6.7

## script/chart2jpg_sample.py
data/sample_chart以下にあるチャートデータをjpgに変換してdata/imgに保存

## script/train.py
data/img以下にあるチャート画像を学習する

## script/train.py
学習したモデルを読み込んでチャート画像をベクトルに変換してcsvファイルに保存


## サンプルコード実行手順
```
cd data
mkdir img
tar -zxvf sample_chart.tar.gz
cd ..
python script/chart2jpg_sample.py
python script/train.py
python script/predict.py
```