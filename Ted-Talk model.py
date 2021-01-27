import pandas as pd
from sklearn.feature_extraction import text
import warnings
import pickle
warnings.filterwarnings("ignore")

df = pd.read_csv('Data/TED.csv')

import datetime

df['film_date'] = df['film_date'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%d-%m-%Y'))
df['published_date'] = df['published_date'].apply(
    lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%d-%m-%Y'))

df['duration_hr'] = df['duration'] / (60 * 60)
df['duration_hr'] = df['duration_hr'].astype(float)
df['duration_hr'] = df['duration_hr'].round(decimals=2)

df_rec = df[['title', 'description']]
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def remove_tags(string):
    result = re.sub('<.*?>', '', string)
    return result


df_rec['description'] = df_rec['description'].apply(lambda cw: remove_tags(cw))

df_rec['description'] = df['description'].str.lower()
df_rec['title'] = df['title'].str.lower()

Text = df_rec['description'].tolist()
tfidf = text.TfidfVectorizer(input=Text, stop_words="english")
matrix = tfidf.fit_transform(Text)

### Get Similarity Scores using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(matrix,matrix)
indices = pd.Series(df_rec['title'])
all_talks=[df_rec['title'][i] for i in range(len(df_rec['title']))]

pickle.dump(cosine_sim, open('cs.pkl','wb'))



