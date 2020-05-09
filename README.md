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
* 入力画像の余白除去でもする？(タイル化にて対応、CVめっちゃいい)
* ckptにfold番号を入れる


## 分かったこと
* 画像サイズは大きい方がいい(256,512,768を試した)
* 入力画像のタイル化は結構いい。でもタイルの作り方とかタイルのデメリットとかは考慮するべき
* avg-poolを[2,2]とか[3,3]とかにすると少し精度は上がった気がするけど学習が少し不安定になる
* 無闇にアンサンブル数を増やすと良くない。多くても10モデルか？？
* そもそも高解像の画像をそのまま入力することが大事ということがわかった（できるだけresizeしない）
    * segmentation->clsだと推論時のGPUメモリとか時間とかがネックになる→seg時でもタイル化を使う？
    * clsだと学習時の入力サイズがネックになる→思い切ってタイルを削るか？


## primary task
* segmentation→classification
    * segmentation→lgbm（segmentationの精度がもっと欲しいところ、パッチで学習した方がいいかも）
    * segmentation→CNN(前にやったのはミスってた)

* もっと大きい画像
    * 徐々に画像サイズを上げていくやつを実装する(後にrestartも検討する)
    * tileの分割数をもう少し増やしてもいいのかも？もう少し余白を少なく
    * Apex対応

* DAの見直し
    * アス比を変えるのも良くない可能性あり、ついにタイル化を採用する機運？

* avgPoolで特徴マップを潰しているのが悪いのではないか説（avgPoolの変更からのhead変更を実装、なんかhead深すぎたのか良くない）
    * 途中までheadだけ学習、その後全体学習(head firstの実装が適当だからそこを整備してから実行)
    * avgPoolをGeMとかにしてみる？
* LossをSmoothL1Lossにしてみる
* 1モデル（アンサンブル無し）を試してみる？

* 画像の背景を黒色化？
* ファイル分けたから次はフォルダ整理する、get_modelの引数を整備
* ttaのアンサンブル方法を変える?（ここはずっと思ってるけどいいのが思いつかないのよな〜）

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

