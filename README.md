# fewshot-LAS

# Calling few shot train module
```
python3 -m tools.train_net --num-gpus 1 --config-file configs/fsucustom-detection/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml
```

# Calling few shot test module
```
python3 -m tools.test_net --num-gpus 1 --config-file configs/fsucustom-detection/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml --eval-only
```
# Calling demo (test)
```
python3 -m demo.demo --config-file /configs/fsucustom-detection/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml --input input1.png --opts MODEL.WEIGHTS /checkpoints/fsucustom/faster_rcnn/faster_rcnn_R_101_FPN_ft_fc_all_1shot/model_final.pth
```
# Calling few shot fine tuning stage 1
```
python3 -m tools.ckpt_surgery --src1 checkpoints/fsucustom/faster_rcnn/faster_rcnn_R_101_FPN_ft_fc_all_1shot/model_final.pth --method randinit --save-dir checkpoints/fsucustom/faster_rcnn/fewshot_fine_tune_stage_1 --coco
```
# Calling few shot fine tuning stage 2 remove last layer
```
python3 -m tools.ckpt_surgery --src1 checkpoints/fsucustom/faster_rcnn/faster_rcnn_R_101_FPN_ft_fc_all_1shot/model_final.pth --method remove --save-dir checkpoints/fsucustom/faster_rcnn/fewshot_fine_tune_stage_1 
```
