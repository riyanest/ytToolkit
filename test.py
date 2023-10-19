
import pandas as pd
from googleapiclient.discovery import build
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiments = SentimentIntensityAnalyzer()

api_key = "AIzaSyCKZ9Y5geIBmjNFgZMhl1-Yh3IX0C1yEpw"

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

def analyze_sentiments(text):
    data=pd.DataFrame(text, columns=['comment','sentiment'])
    sentiments=[]
    for i in data:

        sentiments.append(sentiments.polarity_scores(i['comment']))

    # if sentiment_dict['compound'] >= 0.05 :
    #     return "Positive"
 
    # elif sentiment_dict['compound'] <= - 0.05 :
    #     return "Negative"

    # else :
    #     return "Neutral"
    print(sentiments)



df=video_comments(video_id)
analyze_sentiments(df)