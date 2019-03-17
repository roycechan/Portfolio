from sklearn.metrics import roc_auc_score
import sys
import pandas as pd

def get_auc(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    return auc


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 2:
        print("python score.py <score_csv> <label_csv>")
        exit()
    args = pd.read_csv("output_verification.csv",index_col=False)
    print(args.iloc[:,0])
    score_csv = args[1]
    label_csv = args[2]
    y_score = pd.read_csv(score_csv).score.values
    y_true = pd.read_csv(label_csv).score.values
    auc = get_auc(y_true, y_score)
    print("AUC: ", auc)
