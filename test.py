# Impor
import plotly.graph_objects as go
import os
import flask
import requests
from datetime import datetime

# import google.oauth2.credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import json
# Define API service name and version
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


if __name__ == '__main__':
    # Disable OAuthlib's HTTPs verification when running locally.
    # *DO NOT* leave this option enabled when running in production.
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

# Show plot
    fig.show()
