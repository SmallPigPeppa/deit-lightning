MODEL_NAME=deit_small_patch16_224
python main_pl_scaleLR.py \
    --project DeiT-PL \
    --name ${MODEL_NAME}-REPRO-YD-V \
    --model ${MODEL_NAME} \
    --accelerator npu \
    --devices 8 \
    --batch-size 128 \
    --num_nodes 1 \
    --precision 16 \
    --log_every_n_steps 1 \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename ${MODEL_NAME}\
    --data-path /home/ma-user/work/wenzhuoliu/torch_ds/imagenet
