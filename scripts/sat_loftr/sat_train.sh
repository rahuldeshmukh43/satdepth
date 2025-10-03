#!/bin/bash
# running script
# conda activate satdepth
# nohup ./sat_train.sh &>/dev/null &

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR=$(dirname $(dirname $SCRIPTPATH))

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

TRAIN_IMG_SIZE=448
data_cfg_path="configs/data/satdepth_trainval_${TRAIN_IMG_SIZE}.py"
main_cfg_path="configs/loftr/sat/loftr_ot_dense.py"

yaml_config="$SCRIPTPATH"/"sat_train.yaml"
# logdir=/mnt/cloudNAS4/Rahul/Projects/Models/satloftr/train_satdepth_"$TRAIN_IMG_SIZE"_8_2_with_rotaug_balanced
logdir=/mnt/cloudNAS4/Rahul/Projects/Models/satloftr/train_satdepth_"$TRAIN_IMG_SIZE"_8_2_without_rotaug_balanced

n_nodes=1
n_gpus_per_node=2
torch_num_workers=3

batch_size=3 # per gpu
pin_memory=false # [true| false]
exp_name="satloftr-ot-${TRAIN_IMG_SIZE}-bs=$(($n_gpus_per_node * $n_nodes * $batch_size))"

mkdir -p $logdir

python -u ./train_satloftr.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --config $yaml_config \
    --logdir $logdir \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=100 \
    --flush_logs_every_n_steps=100 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --rot_aug \
    --max_epochs=30 2>&1 | tee "$logdir"/training.stdout