from flask import Flask, render_template, request
import pickle
import re
import pandas as pd

df = pd.read_csv('Data/TED.csv')

df.loc[:, 'title'] = df.loc[:, 'title'].str.lower()

cosine_sim = pickle.load(open('cs.pkl', 'rb'))

app = Flask(__name__)
app._static_folder = ''


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_ted_talk = request.form['ted-talk'].lower()
        user_ted_talk_no_tags = remove_tags(user_ted_talk)
        recommendation = recommend_talks(user_ted_talk_no_tags)
        rec = str(recommendation)[1:-1]
        return render_template('home.html', prediction_text=rec)


def recommend_talks(name):
    indices = pd.Series(df['title'])
    talks = []
    idx = indices[indices == name].index[0]
    sort_index = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top_10 = sort_index.iloc[1:10]
    for i in top_10.index:
        talks.append(indices[i])
    return talks


def remove_tags(string):
    result = re.sub('<.,*?>', '', string)
    return result


if __name__ == '__main__':
    app.run(debug=True)
