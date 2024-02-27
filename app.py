from googleapiclient.discovery import build
from flask import Flask, request, jsonify, render_template
# , redirect, url_for, flash, send_from_directory
import numpy as np
import pandas as pd
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from apiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
import nltk
from nltk.corpus import wordnet
from rembg import remove
from PIL import Image
import os
from werkzeug.utils import secure_filename
import json
# import os
# import re
# import nltk
# from string import punctuation
# from sklearn.preprocessing import LabelEncoder
# from sklearn.utils import resample
# from sklearn.feature_extraction.text import CountVectorizer
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from nltk.corpus import wordnet
# from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
# from youtube_comment_scraper_python import *


# Arguments that need to passed to the build function
DEVELOPER_KEY = "AIzaSyCKZ9Y5geIBmjNFgZMhl1-Yh3IX0C1yEpw"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# creating Youtube Resource Object
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                developerKey=DEVELOPER_KEY)


app = Flask(__name__)
sentiments = SentimentIntensityAnalyzer()


# import pandas as pd

api_key = "AIzaSyCKZ9Y5geIBmjNFgZMhl1-Yh3IX0C1yEpw"


def video_comments(video_id):
    # empty list for storing reply
    df = pd.DataFrame()
    comment = []
    replies = []

    # creating youtube resource object
    youtube = build('youtube', 'v3', developerKey=api_key)

    # retrieve youtube video results
    video_response = youtube.commentThreads().list(
        part='snippet,replies', videoId=video_id).execute()
    #     print(video_response['items'])
    # iterate video response
    while video_response:

        # extracting required info
        # from each result object

        for item in video_response['items']:

            # Extracting comments
            comment.append(item['snippet']['topLevelComment']
                           ['snippet']['textDisplay'])

            # counting number of reply of comment
            replycount = item['snippet']['totalReplyCount']

            # if reply is there
            if replycount > 0:

                # iterate through all reply
                for reply in item['replies']['comments']:

                    # Extract reply
                    reply = reply['snippet']['textDisplay']

                    # Store reply is list
                    replies.append(reply)
                    comment.append(reply)
                    # print(comment,"reply")
                    # df.append(reply)

            # print comment with list of reply

            # print("actual",comment, replies, end = '\n\n')
            # pd.concat([df,comment])
            # print(df,"post append df")

            # empty reply list
            replies = []

        # Again repeat
        if 'nextPageToken' in video_response:
            video_response = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                pageToken=video_response['nextPageToken']
            ).execute()
        else:
            return comment
            break


# Enter video id
video_id = "II1ig6XotTU"

# Call function

df = pd.DataFrame(video_comments(video_id))
# print(df.head())

# Define your sentiment analysis function


def analyze_sentiments(text):
    data = pd.DataFrame(text, columns=['comment'])
    datasentiments = []
    for i in range(len(data)):
        datasentiments.append(
            sentiments.polarity_scores(data.iloc[i]['comment']))

    sentimentdf = pd.DataFrame(datasentiments, columns=[
                               'neg', 'neu', 'pos', 'compound'])
#     print(sentimentdf)
    sentimentdf.drop(['neg', 'neu', 'pos'], axis=1, inplace=True)
    tempsenti = []
    for i in range(len(sentimentdf)):
        if sentimentdf.iloc[i]['compound'] >= 0.05:
            tempsenti.append("Positive")

        elif sentimentdf.iloc[i]['compound'] <= - 0.05:
            tempsenti.append("Negative")
        else:
            tempsenti.append("Neutral")
    sentimentdf['sentiment'] = tempsenti
    sentimentdf.drop(['compound'], axis=1, inplace=True)
    finalcommentdf = data.merge(
        sentimentdf, left_index=True, right_index=True, how='inner')
    # print(finalcommentdf['comment'].tolist())
    comments = finalcommentdf['comment'].tolist()
    sentimentslist = finalcommentdf['sentiment'].tolist()
#     print(finalcommentdf.head())
    json_string = json.dumps(
        {'comments': comments, 'sentiment': sentimentslist})

    return json_string

#     print(datasentiments)


def analyze_sentiment(text):

    sentiment_dict = sentiments.polarity_scores(text)
    if sentiment_dict['compound'] >= 0.05:
        return "Positive"

    elif sentiment_dict['compound'] <= - 0.05:
        return "Negative"

    else:
        return "Neutral"


@app.route("/")
def index():
    return "Hello"


@app.route("/dashboard")
def dashboard():
    message = "Hello, World"
    return render_template('dashboard.html', message=message)


@app.route("/map")
def map():
    message = "Hello, World"
    return render_template('map.html', message=message)


@app.route("/tables")
def tables():
    message = "Hello, World"
    return render_template('tables.html', message=message)


@app.route("/typography")
def typography():
    message = "Hello, World"
    return render_template('typography.html', message=message)


@app.route("/user")
def user():
    message = "Hello, World"
    return render_template('user.html', message=message)


@app.route("/notifications")
def notifications():
    message = "Hello, World"
    return render_template('notifications.html', message=message)
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
        final = analyze_sentiments(data)
        return jsonify({'data': final})
    else:
        return jsonify({'error': 'Missing or invalid data'})


@app.route('/channelStats', methods=['POST'])
def channelstats():
    data = request.json  # Expect JSON data with a 'text' field
    text = data.get('text')

    if text is not None:
        ytrequest = youtube.channels().list(
            part="snippet,contentDetails,statistics", id=text)

        ytresponse = ytrequest.execute()
        return jsonify({'channelData': ytresponse})
    else:
        return jsonify({'error': 'Missing or invalid data'})


@app.route('/videoStats', methods=['POST', 'GET'])
def videostats():  # Expect JSON data with a 'text' field
    id = request.form.get('id')

    if id is not None:
        ytrequest = youtube.videos().list(
            part="snippet,contentDetails,statistics", id=id)
        ytresponse = ytrequest.execute()
        print(ytresponse)
        snippet_details = ytresponse['items'][0]['snippet']

        thumbnail_urls = {}
        thumbnails = snippet_details['thumbnails']
        for size, data in thumbnails.items():
            thumbnail_urls[size] = data['url']
# Storing in a dictionary
        video_details = {
            'title': snippet_details['title'],
            'published_at': snippet_details['publishedAt'],
            'channel_id': snippet_details['channelId'],
            'description': snippet_details['description'],
            'thumbnails': thumbnail_urls,
            'channel_title': snippet_details['channelTitle'],
            'tags': snippet_details['tags'],
            'category_id': snippet_details['categoryId'],
            'live_broadcast_content': snippet_details['liveBroadcastContent'],
            'localized_title': snippet_details['localized']['title'],
            'localized_description': snippet_details['localized']['description'],
            'viewCount': ytresponse['items'][0]['statistics']['viewCount'],
            'likeCount': ytresponse['items'][0]['statistics']['likeCount'],
            'favoriteCount': ytresponse['items'][0]['statistics']['favoriteCount'],
            'commentsCount': ytresponse['items'][0]['statistics']['commentCount'],
            # 'default_audio_language': snippet_details['defaultAudioLanguage']
        }

        # print(ytresponse['items'][0]['statistics']['viewCount']+" testtt")
        return render_template('icons.html', response=video_details)
    else:
        return jsonify({'error': 'Missing or invalid data'})


@app.route('/search', methods=['POST'])
def search():
    data = request.json  # Expect JSON data with a 'text' field
    text = data.get('text')

    if text is not None:
        ytrequest = youtube.search().list(part="snippet", maxResults=25, q=text)
        ytresponse = ytrequest.execute()
        return jsonify({'searchData': ytresponse})
    else:
        return jsonify({'error': 'Missing or invalid data'})


@app.route('/bgrm', methods=['GET', 'POST'])
def bgrm():
    if request.method == 'POST':
        return render_template('brgm.html')
    return render_template('brgm.html')


@app.route('/bgrmsuccess', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        input = Image.open(f)
        # Removing the background from the given Image
        output = remove(input)
        filename = secure_filename(f.filename)
        # Saving the image in the given path
        output.save(os.path.join(
            'D:/Projects/ytToolkit/Sentiment_analysis/static/assets/img', filename), format='png')
        return render_template('acknowledgement.html', name=filename)
        # return send_from_directory('D:/Projects/ytToolkit/Sentiment_analysis/static/assets/img', filename)
        # output.save(output_path)
        # # f.save(f.filename)
        # return render_template(  , name=output.filename)


@app.route('/tagGen', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        title = request.form.get('title')
        tokens = nltk.word_tokenize(title)
    # Perform part-of-speech tagging
        pos_tags = nltk.pos_tag(tokens)
        # Extract all nouns
        nouns = [word for word, pos in pos_tags if pos in [
            'NN', 'NNS', 'NNP', 'NNPS']]

        # Find related terms for each noun
        related_terms = {}
        for noun in nouns:
            synsets = wordnet.synsets(noun)
            related_terms[noun] = set()
            for synset in synsets:
                for lemma in synset.lemmas():
                    related_terms[noun].add(lemma.name())
                    for related_synset in lemma.derivationally_related_forms():
                        print(related_terms[noun].add(related_synset.name()))
        return render_template('tags.html', tags=related_terms)
    return render_template('tagGen.html')


if __name__ == '__main__':
    app.run(debug=True)
