import os
import re
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# 1. Create output directory
output_dir = 'figures'
os.makedirs(output_dir, exist_ok=True)

epochs = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
train_loss = [1.2210, 1.1893, 1.1726, 1.1528, 1.1700, 1.1624, 1.1870, 1.1586, 1.1876, 1.1557]
val_acc = [0.4407, 0.3686, 0.3983, 0.4322, 0.4576, 0.3517, 0.3814, 0.3771, 0.3729, 0.4068]

# 2. Load data
df_txt = pd.read_csv('twitter_data/twitter15/twitter15_source_tweets.txt', sep='\t', names=['tweet_id','text'], dtype=str)
df_lbl = pd.read_csv('twitter_data/twitter15/twitter15_label.txt', sep=':', names=['label','tweet_id'], dtype=str)
df = df_txt.merge(df_lbl, on='tweet_id')

# 3. Clean text
def clean(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^A-Za-z0-9 #]', ' ', text)
    return text.lower()

df['clean_text'] = df['text'].apply(clean)

# 4. Compute sentiment
nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()
df['sentiment'] = df['clean_text'].apply(lambda t: sid.polarity_scores(t)['compound'])

# 5. Label distribution plot (saved)
fig, ax = plt.subplots(figsize=(6,4))
df['label'].value_counts().plot.bar(ax=ax)
ax.set_title('Twitter15 Label Distribution')
ax.set_xlabel('Label')
ax.set_ylabel('Count')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'label_distribution.png'))
plt.close(fig)

# 6. Sentiment histogram plot (saved)
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(df['sentiment'], bins=30, edgecolor='black')
ax.set_title('Sentiment Score Distribution')
ax.set_xlabel('VADER Compound Score')
ax.set_ylabel('Frequency')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
plt.close(fig)

# Plot 1: Training Loss Curve
plt.figure()
plt.plot(epochs, train_loss)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss over Epochs')
plt.tight_layout()
plt.savefig('figures/loss_curve.png')
plt.show()
plt.close()

# Plot 2: Validation Accuracy Curve
plt.figure()
plt.plot(epochs, val_acc)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.tight_layout()
plt.savefig('figures/val_accuracy_curve.png')
plt.show()
plt.close()