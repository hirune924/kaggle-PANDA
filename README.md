# kaggle-PANDA

```
python train_seg_cls.py -dd=../data -ld=../log/ -if=png -smn=resnet34_unet -scd=../ckpt -cmn=se_resnet50 -mt=cat
python train_seg.py -dd=../data -ld=../log/ -if=png -mn=resnet18_unet
python train_cls.py -dd=../data -ld=../log/ -if=png -mn=resnet18 -en=localdebug
python train_cls.py -dd=../data -ld=../log/ -if=png -mn=se_resnet50
python train_cls.py -dd=../data -ld=../log/ -if=png -mn=resnet18 -tl
python train_cls.py -dd=../data -ld=../log/ -if=png -mn=se_resnet50 -hd=custom -is=256
```

## 検証済み
* 5Foldサブミット(256はした,512もした,768もした)
* 画像サイズが大きい方が精度良さそう（CVは確かに上がる、LBも512にして劇的改善したから768にしたらもっと上がった）
* segmentation→classification(うまくいかない)
* 入力画像の余白除去でもする？(タイル化にて対応、CVめっちゃいい)
* ckptにfold番号を入れる

## primary task
* avgPoolで特徴マップを潰しているのが悪いのではないか説（avgPoolの変更からのhead変更を実装、なんかhead深すぎたのか良くない）
    * 途中までheadだけ学習、その後全体学習
    * head薄くしてみる
* 画像の背景を黒色化？
* ファイル分けたから次はフォルダ整理する、get_modelの引数を整備
* まさか、もっと大きい画像使う？？？？？？


## secondary task
* Gleason scoreを考慮してみる？
* そろそろDataAugmentationを考えてみるかモデルを変えてみる？？？
* クラス不均衡への対応？あんまり要らないかも
* 足したいモデル（efficientNet、DenseNet）
* 足したいDataAugmentation(今のところ特に無しkorniaは少し気になる)
* optunaやりたい
* ddpやりたい
* check_pointの名前変更（モデル名入れられる？）
* preds_rounderをもっとクールにする（クラス数可変に）


* スケジューラ（サイクリックにしてみてもいいかも？その場合はearly stopping止めた方がいいと思う）

