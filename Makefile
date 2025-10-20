# 一键复现与训练

install:
	pip install -r requirements.txt

eda:
	python scripts/02_preprocess_and_eda.py

label:
	python scripts/03_label_and_model.py

train-nb:
	python run.py --dataset hotel --model nb --stage train

train-svm:
	python run.py --dataset hotel --model svm --stage train

train-transformer:
	python scripts/train_transformer.py --model_name hfl/chinese-roberta-wwm-ext --train_file data/processed_hotel.csv --test_file data/processed_hotel.csv --text_col tokens_join --label_col sentiment_label --output_dir output/transformer

report:
	python run.py --stage report

all: install eda label train-nb train-svm train-transformer report
