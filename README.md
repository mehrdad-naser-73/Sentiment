# Sentiment

sentiment analysis model for persian food reviews.
This codes uses both ParsBERT and Multilingual BERT for the task of sentiment analysis on persian food reviews.

## Environment
Python 3.7.9, Pytorch 1.7.0, HuggingFace Transformers 4.2.0


## Usage

We have three modes: train, test, webservice

You can train the model with:
```
python sentiment.py --mode train --model_name MODEL_NAME
```
Where MODEL_NAME can be "parsbert" for ParsBERT or "multi" for Multilingual BERT

You can test the model with:
```
python sentiment.py --mode test --load_model_path MODEL_PATH
```
Where MODEL_PATH is the path to your saved model

You can also use a saved model to run a webservice with:
```
python sentiment.py --mode webservice --load_model_path MODEL_PATH
```
Where MODEL_PATH is the path to your saved model. You can use the webservice by sending GET requests to "http://127.0.0.1:5000/" using the following format:
```
{
    "text": "غذا بسیار خوب بود"
}
```
 
