document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('analyzeButton').addEventListener('click', analyzeVideo);
  });
  
  function analyzeVideo() {
    // Get the video URL entered by the user
    const videoURL = document.getElementById('videoURL').value;
  
    // Ensure the URL is from YouTube
    if (!isValidYouTubeURL(videoURL)) {
      alert('Please enter a valid YouTube video URL.');
      return;
    }
  
    // Fetch comments from the YouTube Data API
    fetchYouTubeComments(videoURL)
      .then(comments => {
        // Perform sentiment analysis on the comments
        const sentimentResults = analyzeSentiment(comments);
  
        // Display the sentiment analysis results
        displayResults(sentimentResults);
      })
      .catch(error => { 
        console.error('Error fetching comments or performing sentiment analysis:', error);
        alert('An error occurred while analyzing the video.');
      });
  }
  
  function isValidYouTubeURL(url) {
    // Check if the URL matches a YouTube video URL pattern
    // You may need a more robust URL validation method
    return /^https?:\/\/(?:www\.)?youtube\.com\/watch\?v=/.test(url);
  }
  
  function fetchYouTubeComments(videoURL) {
    // Use the YouTube Data API to fetch comments
    // You'll need to authenticate and make an API request here
    // Example: You can use the fetch() function or a library like axios
  
    // Replace 'YOUR_API_KEY' with your actual YouTube Data API key
    const apiKey = 'AIzaSyAN15fZRcG1Z_6rZoYaEwwH1M4XzTYkzIQ';
    const videoId = extractVideoId(videoURL);
  
    const apiUrl = `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&key=${apiKey}`;
  
    return fetch(apiUrl)
      .then(response => {
        if (!response.ok) {
          throw new Error(`YouTube API request failed: ${response.status} ${response.statusText}`);
        }
        return response.json();
      })
      .then(data => {
        // Extract and return the comments from the API response
        const comments = data.items.map(item => item.snippet.topLevelComment.snippet.textDisplay);
        return comments;
      });
  }
  
  function extractVideoId(url) {
    // Extract the video ID from a YouTube video URL
    const match = url.match(/v=([a-zA-Z0-9_-]+)/);
    return match ? match[1] : null;
  }
  
  function analyzeSentiment(comments) {
    // Use a sentiment analysis library or model to analyze comments
    // Example: You can use Sentiment.js or another library
  

    const { analyze_sentiment } = require('youtube_comments_sentiment_analyser.py'); // Update the path accordingly

// Use the analyze_sentiment function
const textToAnalyze = 'This is a sample text.';
const sentimentScore = analyze_sentiment(textToAnalyze);
console.log(`Sentiment Score: ${sentimentScore}`);

  
    return sentimentResults;
  }
  
  function displayResults(sentimentResults) {
    // Display the sentiment analysis results in the extension's popup.html
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';
  
    sentimentResults.forEach((result, index) => {
      const comment = document.createElement('div');
      comment.innerHTML = `Comment ${index + 1}: ${result.score > 0 ? 'Positive' : result.score < 0 ? 'Negative' : 'Neutral'}`;
      resultsDiv.appendChild(comment);
    });
  }
  