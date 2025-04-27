# Text-Graph Ensemble Method

How to Run:

python3 make_meta.py

python3 fine_tune_bert.py \
  --src twitter_data/twitter15/twitter15_source_tweets.txt \
  --labels twitter_data/twitter15/twitter15_label.txt \
  --out bert_preds.npy

python3 fake_news_gnn.py \
  --src twitter_data/twitter15/twitter15_source_tweets.txt \
  --labels twitter_data/twitter15/twitter15_label.txt \
  --min_df 5 --max_df 0.8 --window 5 \
  --hidden 128 --heads 4 --dropout 0.6 \
  --lr 0.005 --epochs 200

python3 predict_new.py \
  --tweets my_new_tweets.csv \
  --bert_dir models/bert_final \
  --gat_ckpt models/gat_final.pt \
  --ensemble models/ensemble_lr.pkl \
  --le_encoder models/le.pkl \
  --tfidf models/tfidf.pkl

python3 evaluation.py

python accuracy_plot.py --metrics_csv metrics.csv --out accuracy_over_epochs.png
 
