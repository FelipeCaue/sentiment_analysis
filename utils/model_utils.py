import pickle
from sklearn.feature_extraction.text import CountVectorizer
from utils.io_utils import load_config

# BOW_CV = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

config = load_config()

def load_model(path):
    """
        Load LR model
        params:
          @path : path to model
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def load_bow(path):
    """
        Load Matrix of BOW model
        params:
             @path : path to model
    """
    with open(path, "rb") as f:
        bow = pickle.load(f)
    return bow

def get_bag_off_words(text):
    """
        function that transform text in vetctor
        params:
            @text : text to transform
    """
    print(text)
    new_bw = load_bow(config["paths"]["matrix"])
    trasnformed_text = new_bw.transform([text])
    return trasnformed_text


def get_text_sentiment(text):
    """
        function that run model to get  text sentiment
        params:
            text : text to transform
    """
    print(text)
    model = load_model(config["paths"]["model"])
    bow = load_bow(config["paths"]["matrix"])
    trasnformed_text = bow.transform([text])
    proba = model.predict_proba(trasnformed_text)
    idx = proba.argmax()
    sentiment = ''
    if(idx == 0):
        sentiment = 'Negativo'
    else:
        sentiment = "Positivo"
    return {sentiment}
