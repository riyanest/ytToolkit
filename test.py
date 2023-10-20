from apiclient.discovery import build

# Arguments that need to passed to the build function
DEVELOPER_KEY = "AIzaSyCKZ9Y5geIBmjNFgZMhl1-Yh3IX0C1yEpw" 
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# creating Youtube Resource Object
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,developerKey = DEVELOPER_KEY)


request = youtube.videos().list(part="snippet,contentDetails,statistics",id="Ks-_Mh1QhMc")
response = request.execute()
print(response)