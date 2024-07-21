python main_pl.py \
    --project DeiT-PL \
    --name DeiT \
    --model deit_base_patch16_224 \
    --data-path /ppio_net0/torch_ds/imagenet \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True