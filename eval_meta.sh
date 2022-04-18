MODE=$1
DATASET_PATH=$2
GPU_ID=$3
ENCODER_SEED=$4
META_DIR_PATH=$5
MAPPING=$6
TEACHER_PATHS=$7
N=$8


MODEL_SEED=`expr $ENCODER_SEED + 1000`
MODEL_PATH=${META_DIR_PATH}/${MODE}/${ENCODER_SEED}/${MAPPING}/${MODEL_SEED}/Transformer_state_seqlast19.pt
OUT_DIR=${META_DIR_PATH}/${MODE}/comparison


python3 evals/eval_meta.py \
	--model-path ${MODEL_PATH}\
	--len-seq ${N} \
	--gpu ${GPU_ID} \
	--memory ${DATASET_PATH} \
	--out-dir ${OUT_DIR} \
    --teacher-paths ${TEACHER_PATHS}
