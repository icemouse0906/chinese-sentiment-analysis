import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def reliability_diagram(probs, labels, output):
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
    plt.figure(figsize=(5,5))
    plt.plot(prob_pred, prob_true, marker='o', label='Reliability')
    plt.plot([0,1],[0,1],'--',label='Perfect')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.savefig(output)
    print(f'可靠性图已保存到 {output}')

def expected_calibration_error(probs, labels, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    ece = 0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() == 0: continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += abs(acc-conf) * mask.sum() / len(probs)
    return ece

def main():
    parser = argparse.ArgumentParser(description='不确定性与拒识评估')
    parser.add_argument('--probs', type=str, required=True, help='预测概率CSV，需含prob,label列')
    parser.add_argument('--output', type=str, default='output/reliability.png')
    parser.add_argument('--threshold', type=float, default=0.7, help='拒识阈值')
    args = parser.parse_args()
    df = pd.read_csv(args.probs)
    probs = df['prob'].values
    labels = df['label'].values
    reliability_diagram(probs, labels, args.output)
    ece = expected_calibration_error(probs, labels)
    print(f'ECE: {ece:.4f}')
    # 拒识策略
    accept = (probs > args.threshold) | (probs < 1-args.threshold)
    acc = (df['pred']==df['label'])[accept].mean() if accept.sum()>0 else 0
    print(f'拒识后准确率: {acc:.3f}，平均审核量: {1-accept.mean():.3f}')

if __name__ == '__main__':
    main()
