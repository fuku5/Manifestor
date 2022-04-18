DATASET_PATH=$1
SEED=$2
GPU_ID=$3
python3 train/train_e_d_guesser.py --memory ${DATASET_PATH} --seed ${SEED} --gpu ${GPU_ID} --save-prefix encoder
