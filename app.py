from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
# import os
# import re
# import nltk
# from string import punctuation
# from sklearn.preprocessing import LabelEncoder
# from sklearn.utils import resample
# from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from nltk.corpus import wordnet
# from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
# from youtube_comment_scraper_python import *

app = Flask(__name__)
sentiments = SentimentIntensityAnalyzer()


# import pandas as pd
from googleapiclient.discovery import build

api_key = ""

def video_comments(video_id):
    # empty list for storing reply               
    df=pd.DataFrame()
    comment=[]
    replies = []

    # creating youtube resource object
    youtube = build('youtube', 'v3',developerKey=api_key)

    # retrieve youtube video results
    video_response=youtube.commentThreads().list(part='snippet,replies',videoId=video_id).execute()
    #     print(video_response['items'])
    # iterate video response
    while video_response:

        # extracting required info
        # from each result object 
        for item in video_response['items']:
 
            # Extracting comments
            comment.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])

            # counting number of reply of comment
            replycount = item['snippet']['totalReplyCount']

            # if reply is there
            if replycount>0:

                # iterate through all reply
                for reply in item['replies']['comments']:

                    # Extract reply
                    reply = reply['snippet']['textDisplay']

                    # Store reply is list
                    replies.append(reply)
                    comment.append(reply)
                    # print(comment,"reply")
#                     df.append(reply)

            # print comment with list of reply
            
            # print("actual",comment, replies, end = '\n\n')
            # pd.concat([df,comment])
            # print(df,"post append df")

            # empty reply list
            replies = []

        # Again repeat
        if 'nextPageToken' in video_response:
            video_response = youtube.commentThreads().list(
                    part = 'snippet,replies',
                    videoId = video_id,
                    pageToken = video_response['nextPageToken']
                ).execute()
        else:
            return comment
            break

# Enter video id
video_id = "b1UYfLlg4YQ"

# Call function

df=pd.DataFrame(video_comments(video_id))
print(df.head())

# Define your sentiment analysis function
def analyze_sentiments(text):
    data=pd.DataFrame(text, columns=['comment'])
    datasentiments=[]
    for i in range(len(data)):
        datasentiments.append(sentiments.polarity_scores(data.iloc[i]['comment']))
        
    sentimentdf=pd.DataFrame(datasentiments,columns=['neg','neu','pos','compound'])
#     print(sentimentdf)
    sentimentdf.drop(['neg', 'neu','pos'], axis=1, inplace=True)
    tempsenti=[]
    for i in range(len(sentimentdf)):
        if sentimentdf.iloc[i]['compound'] >= 0.05 :
            tempsenti.append("Positive")

        elif sentimentdf.iloc[i]['compound'] <= - 0.05 :
            tempsenti.append("Negative")

        else :
            tempsenti.append("Neutral")
    sentimentdf['sentiment']=tempsenti
    sentimentdf.drop(['compound'],axis=1,inplace=True)
    finalcommentdf = data.merge(sentimentdf, left_index=True, right_index=True, how='inner')

    
#     print(finalcommentdf.head())
    return finalcommentdf.to_json()
    
#     print(datasentiments)
    
def analyze_sentiment(text):

    sentiment_dict = sentiments.polarity_scores(text)
    if sentiment_dict['compound'] >= 0.05 :
        return "Positive"
 
    elif sentiment_dict['compound'] <= - 0.05 :
        return "Negative"

    else :
        return "Neutral"


@app.route("/") 
def index(): 
    return "Hello"

# return sentiment_result

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment_route():
    data = request.json  # Expect JSON data with a 'text' field
    text = data.get('text')

    if text is not None:
        sentiment = analyze_sentiment(text)
        return jsonify({'sentiment': sentiment})
    else:
        return jsonify({'error': 'Missing or invalid data'})
    
@app.route('/analyze_sentiments', methods=['POST'])
def analyze_sentiments_route():
    data = request.json  # Expect JSON data with a 'text' field
    text = data.get('text')

    if text is not None:
        data = video_comments(text)
        final=analyze_sentiments(data)
        return jsonify({'sentiment': final})
    else:
        return jsonify({'error': 'Missing or invalid data'})

if __name__ == '__main__':
    app.run(debug=True)
