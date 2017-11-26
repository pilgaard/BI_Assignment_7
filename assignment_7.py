import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pylab import polyfit, poly1d

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

    x, y, X = pos, neg, np.array([pos])
    X = X.T
    fit = np.polyfit(x,y,deg=1)
    fit_fn = np.poly1d(fit)
    plt.plot(X, y,'ro', X, fit_fn(X), 'b')
    plt.show()
    
    
def run():
    vaderNLTK(HNData)
    print 'done'

run()


