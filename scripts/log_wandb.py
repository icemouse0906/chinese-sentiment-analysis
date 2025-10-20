import wandb
import sys
import os
import json

# 用法：python scripts/log_wandb.py metrics.json
# metrics.json 格式：{"f1":0.85, "accuracy":0.88, ...}

def main():
    if len(sys.argv) < 2:
        print('用法: python scripts/log_wandb.py metrics.json')
        sys.exit(1)
    metrics_file = sys.argv[1]
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    wandb.init(project="sentiment-analysis", name=os.path.basename(metrics_file))
    wandb.log(metrics)
    wandb.finish()

if __name__ == '__main__':
    main()
