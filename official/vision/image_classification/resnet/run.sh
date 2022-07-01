

#!/usr/bin/env bash



rm -rf ResNetGraph

# TF_CPP_VMODULE="xpu_remapper" \
# TF_DUMP_GRAPH_PREFIX="HSResNetGraph" \
TF_CPP_MAX_VLOG_LEVEL=1 \
XPU_GPU_BS=64 \
XPU_ROUND_TRIP=true \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$PYTHONPATH:$PWD:/home/tongsu/models \
python resnet_ctl_imagenet_main.py \
--num_gpus=1 \
--batch_size=64 \
--train_epochs=1 \
--train_steps=10 \
--steps_per_loop=1 \
--log_steps=1 \
--skip_eval \
--use_synthetic_data=true \
--distribution_strategy=off \
--use_tf_while_loop=false \
--use_tf_function=True --enable_xla=False \
--enable_tensorboard=False --enable_checkpoint_and_export=False \
--single_l2_loss_op=True --data_format=channels_last \
--model_dir=/home/tongsu/models/official/vision/image_classification/resnet/model_dir \







# CUDA_VISIBLE_DEVICES=0,1 \
# PYTHONPATH=$PYTHONPATH:$PWD:/home/tongsu/models \
# python resnet_ctl_imagenet_benchmark.py \
# --num_gpus=1 \
# --batch_size=34 \
# --train_epochs=1 \
# --train_steps=10 \
# --steps_per_loop=1 \
# --log_steps=1 \
# --skip_eval \
# --use_synthetic_data=true \
# --distribution_strategy=off \
# --use_tf_while_loop=false \
# --use_tf_function=True --enable_xla=False \
# --enable_tensorboard=False --enable_checkpoint_and_export=False \
# --single_l2_loss_op=True --data_format=channels_last \
# --model_dir=/home/tongsu/models/official/vision/image_classification/resnet/model_dir \
