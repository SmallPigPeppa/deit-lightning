python main_pl.py \
    --project DeiT-PL \
    --name deit_small_patch16_224 \
    --model deit_small_patch16_224 \
    --accelerator npu \
    --batch-size 256 \
    --precision 16 \
    --log_every_n_steps 1 \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True \
    --data-path /home/ma-user/work/dataset/all/torch_ds/imagenet
