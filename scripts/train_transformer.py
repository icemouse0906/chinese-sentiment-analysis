import argparse
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch
import time

def main():
    parser = argparse.ArgumentParser(description='中文预训练模型基线训练')
    parser.add_argument('--model_name', type=str, default='hfl/chinese-roberta-wwm-ext', help='预训练模型名')
    parser.add_argument('--train_file', type=str, required=True, help='训练集CSV文件')
    parser.add_argument('--test_file', type=str, required=True, help='测试集CSV文件')
    parser.add_argument('--text_col', type=str, default='tokens_join', help='文本列名')
    parser.add_argument('--label_col', type=str, default='sentiment_label', help='标签列名')
    parser.add_argument('--output_dir', type=str, default='./output/transformer', help='输出目录')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # 加载数据
    import pandas as pd
    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def preprocess(examples):
        return tokenizer(examples[args.text_col], truncation=True, padding='max_length', max_length=128)
    train_ds = train_ds.map(preprocess, batched=True)
    test_ds = test_ds.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        seed=args.seed,
        logging_dir=args.output_dir,
        report_to=['none'],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )
    start = time.time()
    trainer.train()
    end = time.time()
    print(f"训练时长: {end-start:.1f}秒")
    metrics = trainer.evaluate()
    print(metrics)
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(str(metrics))
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))

if __name__ == '__main__':
    main()
