MODE=$1
MODEL_PATH=${META_DIR_PATH}/${MODE}/${ENCODER_SEED}/${MAPPING}/${MODEL_SEED}/Transformer_state_seqlast19.pt
OUT_DIR=${META_DIR_PATH}/${MODE}/comparison

echo $MODEL_PATH $OUT_DIR $TEACHER_PATHS


python3 evals/eval_meta.py \
	--model-path ${MODEL_PATH}\
	--len-seq ${N} \
	--memory ${DATASET_PATH}\
	--out-dir ${OUT_DIR} \
    --teacher-paths ${TEACHER_PATHS}
