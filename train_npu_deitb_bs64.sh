MODEL_NAME=deit_base_patch16_224
python main_pl.py \
    --project DeiT-PL \
    --name ${MODEL_NAME}-bs64 \
    --model ${MODEL_NAME} \
    --accelerator npu \
    --precision 16 \
    --log_every_n_steps 1 \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename ${MODEL_NAME}-bs64\
    --data-path /home/ma-user/work/wenzhuoliu/torch_ds/imagenet
