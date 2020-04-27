# kaggle-PANDA


'''
python train.py -dd=../data -ld=../log/ -if=png -mn=se_resnet50
'''

* モデル足す
* DataAugmentation足す

* 5Fold実装
* early stoppingを変更
* check_pointの名前変更（特にstepに）
* LRをLogging
    * 一応できた。複数Logger対応はまだ()
* LossをRMSEにしていろんなmonitoringもそれにする
    * 完了
* スケジューラも少し変更
    * いったん泳がす？
* 
