MODEL_NAME=deit_small_patch16_224
#MODEL_NAME=deit_base_patch16_224

python main_pl_scaleLR_ra_taylor.py \
    --epochs 5 \
    --lr 5e-4 \
    --warmup-lr 5e-4 \
    --min-lr 5e-4 \
    --seed 3407 \
    --project DeiT-PL-Taylor \
    --name ${MODEL_NAME}-REPRO-YD-RA \
    --model ${MODEL_NAME} \
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
    --data-path /home/ma-user/work/wenzhuoliu/torch_ds/imagenet
