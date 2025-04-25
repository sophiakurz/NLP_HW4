import pandas as pd

# load your original TSV
df = pd.read_csv(
    'twitter_data/twitter15/twitter15_source_tweets.txt',
    sep='\t',
    names=['tweet_id','text'],
    dtype=str
)

# write out CSV with header
df.to_csv('my_new_tweets.csv', index=False)
print("Wrote my_new_tweets.csv with columns:", df.columns.tolist())
