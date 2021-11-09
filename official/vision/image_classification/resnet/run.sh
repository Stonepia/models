#!/usr/bin/env bash



rm -rf ResNetGraph

TF_DUMP_GRAPH_PREFIX="ResNetGraph" \
TF_CPP_VMODULE="xpu_remapper" \
TF_CPP_MAX_VLOG_LEVEL=4 \
PYTHONPATH=$PYTHONPATH:$PWD:/home/tongsu/models \
python resnet_ctl_imagenet_main.py \
--num_gpus=1 \
--batch_size=32 \
--train_epochs=2 \
--train_steps=20 \
--skip_eval=True \
--use_synthetic_data=true \
--distribution_strategy=off \
--log_steps=1 \
--enable_tensorboard=True \
--enable_checkpoint_and_export=False \
--use_tf_function=True \
--enable_xla=False \
--model_dir=/home/tongsu/models/official/vision/image_classification/resnet/model_dir \
--use_tf_while_loop=False
# --profile_steps=3,5 \
# --data_format=channels_last 
# --dtype=bf16 




# python resnet_ctl_imagenet_main.py \
#     --num_gpus=1 \
#     --batch_size=32 \
#     --train_epochs=1 \
#     --train_steps=60 \
#     --use_synthetic_data=true \
#     --data_dir=/home/tongsu/Datasets/ImageNetTiny \
#     --skip_eval=true \
#     --use_tf_function=True 
