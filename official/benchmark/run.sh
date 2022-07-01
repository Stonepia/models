# CUDA_VISIBLE_DEVICES=0,1 \
# TF_DUMP_GRAPH_PREFIX="XResNetGraph" \
# TF_CPP_VMODULE="xpu_remapper" \
# TF_CPP_MAX_VLOG_LEVEL=4 \
PYTHONPATH=$PYTHONPATH:$PWD:/home/tongsu/models \
python resnet_ctl_imagenet_benchmark.py --profile_steps=15,18
