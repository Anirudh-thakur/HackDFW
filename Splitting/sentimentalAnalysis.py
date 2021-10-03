import pandas as pd
import numpy as np

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


df = pd.read_excel("ReelsData.xlsx")
print(df.head())
desc_blob = [TextBlob(desc) for desc in df['comment']]
#add the sentiment metrics to the dataframe
df['tb_Pol'] = [b.sentiment.polarity for b in desc_blob]
df['tb_Subj'] = [b.sentiment.subjectivity for b in desc_blob]
#load VADER
analyzer = SentimentIntensityAnalyzer()
#Add VADER metrics to dataframe
df['compound'] = [analyzer.polarity_scores(
    v)['compound'] for v in df['comment']]
df['neg'] = [analyzer.polarity_scores(v)['neg'] for v in df['comment']]
df['neu'] = [analyzer.polarity_scores(v)['neu'] for v in df['comment']]
df['pos'] = [analyzer.polarity_scores(v)['pos'] for v in df['comment']]
df.head(3)
