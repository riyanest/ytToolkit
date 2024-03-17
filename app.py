from googleapiclient.discovery import build
from flask import Flask, request, jsonify, render_template
# , redirect, url_for, flash, send_from_directory
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import wordnet
from rembg import remove
from PIL import Image
import os
from werkzeug.utils import secure_filename

import os
import flask
import requests
from datetime import datetime

# import google.oauth2.credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
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


# Arguments that need to passed to the buil

API_SERVICE_NAME = "youtubeAnalytics"
API_VERSION = "v2"
CLIENT_SECRETS_FILE = 'D:/Projects/ytToolkit/Sentiment_analysis/client_secret_361851587388-5dmhhj1ctokd64u7stbff0g7ctg0j932.apps.googleusercontent.com.json'
# Load your downloaded JSON credentials file

# 361851587388-5dmhhj1ctokd64u7stbff0g7ctg0j932.apps.googleusercontent.com

SCOPES = ['https://www.googleapis.com/auth/yt-analytics.readonly']


def get_service():
    flow = InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, SCOPES)
    credentials = flow.run_local_server()
    return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)


def execute_api_request(client_library_function, **kwargs):
    response = client_library_function(**kwargs).execute()
    print(response)
    return response


today = datetime.today().strftime('%Y-%m-%d')


app = Flask(__name__)
sentiments = SentimentIntensityAnalyzer()

api_key = "AIzaSyCKZ9Y5geIBmjNFgZMhl1-Yh3IX0C1yEpw"

# import pandas as pd
youtube = build('youtube', 'v3', developerKey=api_key)


def video_comments(video_id):
    # empty list for storing reply
    df = pd.DataFrame()
    comment = []
    replies = []

    # creating youtube resource object

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


video_details = {
    'videoStats': "none",
    'channelStats': "none",
    'bg': "none",
    'sentiments': {},
    'tags': []
}

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
    Finalsentiments = {}
    # print(finalcommentdf['comment'][0])
    comments = finalcommentdf['comment'].tolist()
    sentimentslist = finalcommentdf['sentiment'].tolist()
    for i in range(0, len(finalcommentdf)):
        # Finalsentiments.append((finalcommentdf['comment'][i]: finalcommentdf['sentiment'][i]))
        Finalsentiments[comments[i]] = sentimentslist[i]

    print(finalcommentdf)
    # json_string = {'comments': comments, 'sentiment': sentimentslist}

    return Finalsentiments

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
    df = pd.read_csv(
        "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
    df.columns = [col.replace("AAPL.", "") for col in df.columns]

    # Create figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=list(df.Date), y=list(df.High)))

    # Set title
    fig.update_layout(
        title_text="Time series with range slider and selectors"
    )

    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    graphJSON = fig.to_json()

    return render_template('index.html', graphJSON=graphJSON)


@app.route("/dashboard")
def dashboard():
    message = "Hello, World"
    return render_template('dashboard.html', response=video_details)


@app.route('/channelStats', methods=['POST'])
def channelstats():
    data = request.form.get('id')  # Expect JSON data with a 'text' field

    if data is not None:
        ytrequest = youtube.channels().list(
            part="snippet,contentDetails,statistics", id=data)

        video_details['channelStats'] = ytrequest.execute()[
            'items'][0]['snippet']
        print(video_details['channelStats'])
        # return render_template('channelData.html', response=video_details)
    else:
        return jsonify({'error': 'Missing or invalid data'})


@app.route('/videoStats', methods=['POST', 'GET'])
def videostats():  # Expect JSON data with a 'text' field+
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
        video_details['videoStats'] = {
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

        data = video_comments(id)
        video_details['sentiments'] = analyze_sentiments(data)

        # print(ytresponse['items'][0]['statistics']['viewCount']+" testtt")
        return render_template('dashboard.html', response=video_details)
    else:
        return render_template('dashboard.html', response="none")


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
        f = request.files['file']
        input = Image.open(f)
        # Removing the background from the given Image
        output = remove(input)
        filename = secure_filename(f.filename)
        # Saving the image in the given path
        output.save(os.path.join(
            'D:/Projects/ytToolkit/Sentiment_analysis/static/assets/img', filename), format='png')
        video_details['bg'] = filename
        return render_template('dashboard.html', response=video_details)
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
        video_details['tags'] = related_terms
        return render_template('dashboard.html', response=video_details)
    return render_template('dashboard.html')


@app.route('/analytics', methods=['GET'])
def analytics():  # Expect JSON data with a 'text' field
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

    youtubeAnalytics = get_service()
    response = execute_api_request(
        youtubeAnalytics.reports().query,
        ids='channel==MINE',
        startDate='2018-01-01',
        endDate='2024-03-01',
        metrics='estimatedMinutesWatched,views,likes,subscribersGained',
        dimensions='month')

    dates = []
    minutesWatched = []
    views = []
    likes = []
    subs = []
    for response in response['rows']:
        dates.append(response[0])
        minutesWatched.append(response[1])
        views.append(response[2])
        likes.append(response[3])
        subs.append(response[4])

# Create traces
    traces = []

    traces.append(go.Scatter(x=dates, y=minutesWatched,
                  mode='lines+markers', name="MINUTES WATCHED"))
    traces.append(go.Scatter(x=dates, y=views,
                  mode='lines+markers', name="VIEWS"))
    traces.append(go.Scatter(x=dates, y=likes,
                  mode='lines+markers', name="LIKES"))
    traces.append(go.Scatter(x=dates, y=subs,
                  mode='lines+markers', name="DATES"))

# Create figure
    fig = go.Figure(data=traces)

# Update layout
    fig.update_layout(
        title='Metrics Over Time',
        xaxis=dict(
            title='Month',
            type='date',
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            )
        ),
        yaxis=dict(
            title='Count'
        )
    )

    # fig.show()
    graphJSON = fig.to_json()
    video_details['graphJSON'] = graphJSON
    return render_template('dashboard.html', response=video_details)

    # return jsonify({'response': response})
    # for response in response['columnHeaders']:
#     #     print(response['name'])


#     # Create figure

# # Show plot
#     fig.show()
#     return jsonify({'month': dates,
#                     "minutesWatched": minutesWatched,
#                     "views": views,
#                     "likes": likes,
#                     "subs": subs})


if __name__ == '__main__':
    app.run(debug=True)
