import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# 1) Read your predictions
preds = pd.read_csv('new_predictions.csv', dtype=str)[['tweet_id','pred_label']]

# 2) Read the ground‑truth labels
#    twitter15_label.txt is in the form label:tweet_id, one per line
labels = pd.read_csv(
    'twitter_data/twitter15/twitter15_label.txt',
    sep=':',
    names=['label','tweet_id'],
    dtype=str
)

# 3) Merge on tweet_id
df = preds.merge(labels, on='tweet_id', how='inner')

# 4) Compute and print metrics
print("Overall accuracy: ", accuracy_score(df['label'], df['pred_label']))
print("\nFull classification report:\n")
print(classification_report(df['label'], df['pred_label'], digits=4))