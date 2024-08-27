MODEL_NAME=deit_small_patch16_224
python main_pl_scaleLR_test.py \
    --project DeiT-PL-test \
    --name ${MODEL_NAME}-REPRO \
    --model ${MODEL_NAME} \
    --accelerator npu \
    --devices 8 \
    --num_nodes 1 \
    --precision 16 \
    --log_every_n_steps 1 \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename ${MODEL_NAME}\
    --data-path /home/ma-user/work/dataset/all/torch_ds/imagenet \
    --ckpt_path ckpt/deit_small_patch16_224-v4.ckpt
