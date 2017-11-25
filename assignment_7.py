import numpy as np
import pandas as pd
import nltk
import heapq
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Uncomment if first time using Vader.
#nltk.download('vader_lexicon')
dataSet = "data/hn_items.csv"
HNData = pd.read_csv(dataSet)

def vaderNLTK(data):
    model = SentimentIntensityAnalyzer()
    data = pd.DataFrame(data)
    df = data.dropna(subset=["text"])
    pos = []
    neg = []
    index = 0
    print 'running sentiment analysis..'
    for row in df.iterrows():
        text = df.iloc[index]["text"]
        res = model.polarity_scores(text)
        pos.append(res['pos'])
        neg.append(res['neg'])
        index = index+1   
    print 'sentiment analysis finished'
    df['pos'] = pos
    df['neg'] = neg
    best = df.nlargest(5, 'pos')
    worst = df.nlargest(5, 'neg')     
    print 'Best: ', '\n', best[['text', 'pos']]
    print 'Worst: ', '\n', worst[['text', 'neg']]
    
def run():
    vaderNLTK(HNData)
    print 'done'

run()


