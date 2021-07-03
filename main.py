from fastapi import FastAPI
from utils.io_utils import load_config
from utils.model_utils import load_model, load_bow, get_text_sentiment
import uvicorn
config = load_config()
model = load_model(config["paths"]["model"])
bow = load_bow(config["paths"]["matrix"])

app = FastAPI()


@app.get("/")
def read_root():
    return "Up running"

@app.get("/predict/")
def predict(text):
    sentimet = get_text_sentiment(text)
    return {"sentimet": sentimet}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)