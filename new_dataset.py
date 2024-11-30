import pandas as pd
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import nltk

nltk.download('punkt')

df_category = pd.read_csv('corpus/df_file.csv')
df_emotion = pd.read_parquet('corpus/train-00000-of-00001.parquet')

def split_paragraphs_to_sentences(paragraph):
    return sent_tokenize(paragraph)

df_category['sentences'] = df_category['Text'].apply(split_paragraphs_to_sentences)
all_sentences = df_category['sentences'].explode().tolist()

emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
category_model = pipeline("text-classification", model="cardiffnlp/tweet-topic-21-multi")

predicted_emotions = [emotion_model(sentence)[0]['label'] for sentence in all_sentences]
predicted_categories = [category_model(sentence)[0]['label'] for sentence in all_sentences]

final_df = pd.DataFrame({
    'text': all_sentences,
    'category': predicted_categories,
    'emotion': predicted_emotions
})

final_df.to_parquet('data_combined.parquet', index=False)
