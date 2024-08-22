MODEL_NAME=deit_base_patch16_224
python main_pl.py \
    --project DeiT-PL \
    --name ${MODEL_NAME}-DEBUG \
    --model ${MODEL_NAME} \
    --offline \
    --accelerator npu \
    --num_nodes ${NNODES} \
    --lr 3e-3 \
    --batch-size 256 \
    --precision 16 \
    --log_every_n_steps 1 \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename ${MODEL_NAME}\
    --data-path /opt/huawei/dataset/all/torch_ds/imagenet
