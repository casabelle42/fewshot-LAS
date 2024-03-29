# fewshot-LAS

# Calling few shot train module
```
OLD : python3 -m tools.train_net --num-gpus 1 --config-file configs/fsucustom-detection/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml
NEW : python3 -m tools.train_net --num-gpus 1 --config-file configs/fsucustom-detection/faster_rcnn_R_101_FPN_base.yaml
```

# Calling few shot test module
```
python3 -m tools.test_net --num-gpus 1 --config-file configs/fsucustom-detection/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml --eval-only
```
# Calling demo (test)
```
python3 -m demo.demo --config-file configs/fsu/all_1shot.yaml --input input1.jpg --output save --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/t1_faster_rcnn_R_101_FPN_final.pth
```
# Calling few shot fine tuning stage 1
```
Old : python3 -m tools.ckpt_surgery --src1 checkpoints/fsucustom/faster_rcnn/faster_rcnn_R_101_FPN_ft_fc_all_1shot/model_final.pth --method randinit --save-dir checkpoints/fsucustom/faster_rcnn/fewshot_fine_tune_stage_1 --coco
New : python3 -m tools.ckpt_surgery --src1 checkpoints/fsucustom/faster_rcnn/t1_faster_rcnn_R_101_FPN_base/model_final.pth --method randinit --save-dir checkpoints/fsucustom/faster_rcnn/fewshot_fine_tune_stage_1 --coco
```
# Calling few shot fine tuning stage 2 remove last layer
```
Old : python3 -m tools.ckpt_surgery --src1 checkpoints/fsucustom/faster_rcnn/faster_rcnn_R_101_FPN_ft_fc_all_1shot/model_final.pth --method remove --save-dir checkpoints/fsucustom/faster_rcnn/fewshot_fine_tune_stage_1 
New : python3 -m tools.ckpt_surgery --src1 checkpoints/fsucustom/faster_rcnn/t1_faster_rcnn_R_101_FPN_base/model_final.pth --method remove --save-dir checkpoints/fsucustom/faster_rcnn/fewshot_fine_tune_stage_1 
```
# Calling few shot fine tuning stage 3 fine tune pred on novel set config
```
Old : python3 -m tools.train_net --num-gpus 1 --config-file configs/fsucustom-detection/novel_train_last_layer.yaml --opts MODEL.WEIGHTS checkpoints/fsucustom/faster_rcnn/fewshot_fine_tune_stage_1/model_reset_remove.pth
New : python3 -m tools.train_net --num-gpus 4 --config-file configs/fsucustom-detection/novel_train_last_layer.yaml --opts MODEL.WEIGHTS checkpoints/fsucustom/faster_rcnn/fewshot_fine_tune_stage_1/model_reset_remove.pth
```
# Combining layers
```
python3 -m tools.ckpt_surgery --src1 checkpoints/fsucustom/faster_rcnn/faster_rcnn_R_101_FPN_ft_fc_all_1shot/model_final.pth --src2 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/model_final.pth --method combine --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_combined --coco
python3 -m tools.ckpt_surgery --src1 checkpoints/fsucustom/faster_rcnn/t1_faster_rcnn_R_101_FPN_base/model_final.pth --src2 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/model_final.pth --method combine --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_combined --coco
```
# Train on balanced dataset
```
old : python3 -m tools.train_net --num-gpus 8 --config-file configs/fsu/all_1shot.yaml --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_combined/model_reset_combine.pth
new : python3 -m tools.train_net --num-gpus 8 --config-file configs/fsu/all_1shot.yaml --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_combined/model_reset_combine.pth
