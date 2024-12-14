# youtube_comment_fetcher.py
import os
import time
from googleapiclient.errors import HttpError
import random
from googleapiclient.discovery import build
from sentiment_model import analyze_sentiment, generate_pie_chart, save_results_to_csv, display_comment_statistics

# YouTube API key
API_KEY = 'AIzaSyBFuESZp85xp94hU-rgh9DaomJ4vSLzA28'  # Replace with your actual YouTube API key
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_live_chat_id(video_id):
    try:
        video_response = youtube.videos().list(
            part='liveStreamingDetails',
            id=video_id
        ).execute()

        if 'items' in video_response and video_response['items']:
            live_chat_id = video_response['items'][0]['liveStreamingDetails']['activeLiveChatId']
            return live_chat_id
        else:
            return None
    except Exception as e:
        print(f"An error occurred while fetching live chat ID: {e}")
        return None

def get_live_chat_comments(video_id, max_comments=2000):
    comments = []
    next_page_token = None
    live_chat_id = get_live_chat_id(video_id)

    if not live_chat_id:
        print("This video is not a live stream or the live chat is not available.")
        return comments

    while len(comments) < max_comments:
        try:
            request = youtube.liveChatMessages().list(
                liveChatId=live_chat_id,
                part="snippet,authorDetails",
                maxResults=2000,
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response['items']:
                comment = item['snippet']['displayMessage']
                author = item['authorDetails']['displayName']
                comments.append((comment, author))

                if len(comments) >= max_comments:
                    break

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

            time.sleep(3)  # Increased delay to avoid hitting API rate limits
        except HttpError as e:
            print(f"An HTTP error occurred: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

    return comments

def get_comments(video_id):
    """Fetch comments for a given YouTube video ID."""
    comments = []
    next_page_token = None
    max_comments = 100000  # Set the upper limit for the number of comments
    count = 0
    
    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,  # Fetch the maximum 100 comments per request
            pageToken=next_page_token,
            order="relevance"
        )
        response = request.execute()
        time.sleep(3)  # Adding a 1-second delay after each API request
        
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
            likes = item['snippet']['topLevelComment']['snippet']['likeCount']
            truncated_comment = comment[:512]  # Truncate comment to a maximum of 512 characters
            comments.append((truncated_comment, likes))
            count += 1
            
            if count >= max_comments:
                break
        
        if 'nextPageToken' in response and count < max_comments:
            next_page_token = response['nextPageToken']
        else:
            break

    print(f"Total comments fetched: {len(comments)}")
    return comments


def main(video_id):
    """Main function to fetch comments and perform sentiment analysis."""
    comments = get_comments(video_id)
    if comments:
        # Extracting all comments and likes
        comment_texts = [comment for comment, _ in comments]  
        comment_likes = [likes for _, likes in comments]
        
        if comment_texts:
            sentiments = []
            sentiment_values = []
            sentiment_counts = {
                'Very Positive': 0,
                'Positive': 0,
                'Neutral': 0,
                'Negative': 0,
                'Very Negative': 0
            }
            
            # Analyze sentiment for each comment
            for comment_text in comment_texts:
                sentiment, sentiment_value = analyze_sentiment(comment_text)
                sentiments.append(sentiment)
                sentiment_values.append(sentiment_value)
                # Update sentiment counts
                sentiment_counts[sentiment] += 1
            
            # Display statistics
            total_comments = len(comment_texts)
            display_comment_statistics(total_comments, sentiment_counts)
        

            # Save results
            save_results_to_csv(video_id, comment_texts, sentiments, sentiment_values, comment_likes)

            # Generate charts
            generate_pie_chart(sentiment_counts)
        else:
            print("No comments found for analysis.")
    else:
        print("No comments found for this video.")

if __name__ == "__main__":
    video_id = input("Enter the YouTube video ID: ")
    main(video_id)
