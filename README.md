# kaggle-PANDA


'''
python train.py -dd=../data -ld=../log/ -if=png -mn=resnet18 -en=localdebug
python train.py -dd=../data -ld=../log/ -if=png -mn=se_resnet50
'''

## primary task
* 5Foldサブミット(256はした,512もした)
* 画像サイズが大きい方が精度良さそう（CVは確かに上がる、LBも512にして劇的改善したから768にしてどうなるか確認）
* そろそろファイル分ける
* segmentation→classification

## secondary task
* 足したいモデル（efficientNet、DenseNet）
* 足したいDataAugmentation(今のところ特に無しkorniaは少し気になる)

* check_pointの名前変更（モデル名入れられる？）
* preds_rounderをもっとクールにする（クラス数可変に）


* スケジューラ（サイクリックにしてみてもいいかも？その場合はearly stopping止めた方がいいと思う）

