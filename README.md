# fewshot-LAS

# Calling few shot train module
```
python3 -m tools.train_net --num-gpus 1 --config-file configs/fsucustom-detection/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml
```

#Calling few shot test module
```
python3 -m tools.test_net --num-gpus 1 --config-file configs/fsucustom-detection/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml --eval-only
```
