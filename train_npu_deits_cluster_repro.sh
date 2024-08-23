MODEL_NAME=deit_small_patch16_224
python main_pl_scaleLR.py \
    --project DeiT-PL \
    --name ${MODEL_NAME}-REPRO \
    --model ${MODEL_NAME} \
    --offline \
    --accelerator npu \
    --devices ${NGPUS_PER_NODE} \
    --num_nodes ${NNODES} \
    --precision 16 \
    --sync_batchnorm \
    --log_every_n_steps 1 \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename ${MODEL_NAME}\
    --data-path /opt/huawei/dataset/all/torch_ds/imagenet

