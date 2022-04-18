# Manifestor
The implementation of Manifestor and the other models for experiments.

## Quickstart
- Download the datasets and place them to the following directory:
  - Unskilled dataset: https://drive.google.com/file/d/1lGoWOdZ4FFwW6Hi7V5sM8Hrf2gNhXOHB/view?usp=sharing
  - Skilled dataset: https://drive.google.com/file/d/146k2zVvUW1ghD3JvMVpLQchonxezbCeH/view?usp=sharing
  - To: ./data/records/
- Run jupyter and open main.ipynb
```bash
pipenv install
pipenv shell
jupyter lab
```

## To develop a dataset by yourself
1. Train A2C agent
```bash
python3  a2c_gym.py --gpu 0 --num-env 80 --update-steps 5 --eval-n-runs 50 --n-hidden-size 512 --max-grad-norm 0.5
```

2. Make a dataset
```bash
A2C_PATH={{ the path for the A2C agent you trained }}
DATASET_NAME={{ your dataset name }}
python3  a2c_gym.py
	--demo \
	--load ${A2C_PATH} \
	--record ${RECORD_NAME} \
	--eval-n-runs 3000 \
	--num-envs 80 \
	--n-hidden-size 512 \
	--gpu $GPU_ID
```

3. Change parameters in main.ipynb
```python3
DATASET_PATH ='./data/records/{{ your dataset name }}.pickle.gzip'
```
