import pandas as pd
import numpy as np
import re
import nltk
import mlflow
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


nltk.download('stopwords')
df = pd.read_csv('../data/processed/training.1600000.processed.noemoticon.csv', encoding='latin',header = None)
df.columns = ['target','ids','Date','flag','user','text']

def cleaned_text(text):
    # eliminating urls
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',text)
    # eliminating mentions
    text = re.sub('@[^\s]+',' ', text)
    # eliminating
    text = re.sub('[\s]+', ' ', text)
    # eliminating hashtags
    text = re.sub(r'#([^\s]+)', r' ', text)
    return text

df['tidy_tweet'] = df.text.apply(cleaned_text)
df.loc[df['target'] == 4, 'target'] = 1
df = df.drop(columns = ['ids','Date','flag','user'])

StopWords = set(stopwords.words('english'))

def text_cleaner(text):
    text = text.str.replace('[^a-zA-Z#]',' ')
    text = text.str.lower()
    text = text.apply(lambda x: ' '.join(w for w in x.split() if w not in StopWords))
    return text
df['tidy_tweet'] = text_cleaner(df['tidy_tweet'])

tokenized_tweet = df['tidy_tweet'].apply(lambda x: x.split())


stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

df['tidy_tweet'] = tokenized_tweet


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#BOW_CV= CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
BOW_CV= TfidfVectorizer(max_df=0.95, min_df=1, max_features=2000, stop_words='english')
# bag-of-words feature matrix
bow = BOW_CV.fit_transform(df['tidy_tweet'])

xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(bow, df['target'], random_state=42, test_size=0.3)


mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('sentiment_analysis')
mlflow.sklearn.autolog(log_models=True)

with mlflow.start_run() as run:
    lreg = LogisticRegression()
    lreg.fit(xtrain_bow, ytrain) # training the model
    prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
    prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
    prediction_int = prediction_int.astype(np.int)
    accuracy = lreg.score(xvalid_bow, yvalid)
    mlflow.sklearn.log_model(lreg,'lreg')
    class_report = classification_report(yvalid, prediction_int)