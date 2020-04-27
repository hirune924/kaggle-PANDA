# kaggle-PANDA


'''
python train.py -dd=../data -ld=../log/ -if=png -mn=se_resnet50
'''

## primary task
* resized Dataの修正
* 5Fold実装（実行、logファイル名、サブミットについて）
* segmentation→classification
* そろそろファイル分ける
* 画像サイズが大きい方が精度良さそう

## secondary task
* 足したいモデル（efficientNet、DenseNet）
* 足したいDataAugmentation(今のところ特に無し)

* check_pointの名前変更（モデル名入れられる？）
* preds_rounderをもっとクールにする（クラス数可変に）


* スケジューラ（サイクリックにしてみてもいいかも？その場合はearly stopping止めた方がいいと思う）

