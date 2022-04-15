MODE=$1


if [ "${MODE}" = "optimal" ]; then
	label_mode=g0
elif [ "${MODE}" = "ablation" ]; then
	label_mode=g100
elif [ "${MODE}" = "Manifestor" ]; then
	label_mode=g2
else
	echo invalid target model
	exit
fi

ENCODER_PATH=${ENCODER_DIR_PATH}/${ENCODER_SEED}/EncoderDecoder19.pt
MODEL_SEED=`expr $ENCODER_SEED + 1000`

python3 train/train_e_d_g_meta.py \
	--memory ${DATASET_PATH} \
	--seed ${MODEL_SEED} \
	--translater $MAPPING \
	--gpu $GPU_ID \
	--e-d-path ${ENCODER_PATH} \
	--save-prefix $MODE/${ENCODER_SEED} \
	--len-seq $N \
	--label-mode ${label_mode}

if [ "${MODE}" = "Manifestor" ]; then
	MODEL_PATH=${META_DIR_PATH}/$MODE/${ENCODER_SEED}/${MAPPING}/${MODEL_SEED}/Transformer_state_seqlast19.pt
	python3 evals/eval_e_d_guesser_goal_loss.py \
		--model-path ${MODEL_PATH}\
		--guesser-path ${ENCODER_PATH}\
		--memory ${DATASET_PATH}\
		--translater ${MAPPING}\
		--report-name ${ENCODER_SEED}_${MAPPING}.json 
fi
