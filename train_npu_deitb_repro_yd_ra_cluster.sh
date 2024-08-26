MODEL_NAME=deit_base_patch16_224
python main_pl_scaleLR_ra.py \
    --seed 3407 \
    --project DeiT-PL \
    --name ${MODEL_NAME}-REPRO-YD-RA \
    --model ${MODEL_NAME} \
    --offline \
    --batch-size 128 \
    --trainer.accelerator npu \
    --trainer.devices 8 \
    --trainer.num_nodes 1 \
    --trainer.precision 16 \
    --trainer.log_every_n_steps 1 \
    --lr_monitor.logging_interval epoch \
    --model_checkpoint.dirpath ckpt \
    --model_checkpoint.save_weights_only True \
    --model_checkpoint.filename ${MODEL_NAME}\
    --data-path /opt/huawei/dataset/all/torch_ds/imagenet
