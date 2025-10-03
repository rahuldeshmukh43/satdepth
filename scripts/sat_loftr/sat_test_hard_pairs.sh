#!/bin/bash
# running script
# Usage:
# conda activate loftr
# nohup ./sat_ot_test.sh <testing-set-name> <<test pairlist csv> &>/dev/null &

if [ -z "$1" ];
then
	# is empty
	testing_foldername=testing_set
else
	testing_foldername=$1
fi

test_pairlist=$2

SCRIPTPATH=$(dirname $(readlink -f "$0"))
#PROJECT_DIR="${SCRIPTPATH}/../../"
PROJECT_DIR=$(dirname $(dirname $SCRIPTPATH))

#conda activate deepkp_cuda_11
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

TRAIN_IMG_SIZE=448
data_cfg_path="configs/data/satdepth_trainval_${TRAIN_IMG_SIZE}.py"
main_cfg_path="configs/loftr/outdoor/loftr_ot_dense.py"
yaml_config="$SCRIPTPATH"/"sat_test.yaml"

outdir=/mnt/cloudNAS4/Rahul/Projects/Models/loftr/train_satdepth_448_8_2_with_rotaug_balanced
# outdir=/mnt/cloudNAS4/Rahul/Projects/Models/loftr/train_satdepth_448_8_2_without_rotaug_balanced

batch_size=16 # [8: 1080ti, ?: a5000]
device=0

mkdir -p $outdir

python -u ./sat_test.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --config $yaml_config \
    --test_pairlist $test_pairlist \
    --outdir $outdir \
    --device $device \
    --testing_foldername $testing_foldername \
    --do_match_plot \
    --batch_size=${batch_size} | tee "$outdir"/"$testing_foldername".stdout

