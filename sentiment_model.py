import time
import csv
import random
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, RobertaTokenizer, BertTokenizer, BertModel
from wordcloud import WordCloud
import torch
import torch.nn as nn

# Initialize VADER and RoBERTa models
vader_analyzer = SentimentIntensityAnalyzer()
roberta_analyzer = pipeline("sentiment-analysis")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Load BERT tokenizer and model for CNN-RNN
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Define the CNN-RNN Model
class BertCNNRNN(nn.Module):
    def __init__(self, output_size):
        super(BertCNNRNN, self).__init__()
        self.bert = bert_model
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        with torch.no_grad():
            x = self.bert(x)[0]  # Get the last hidden states
        x = x.permute(0, 2, 1)  # Change shape for Conv1d
        x = torch.relu(self.conv1(x))
        x = x.permute(0, 2, 1)  # Change shape for LSTM
        x, (hn, cn) = self.lstm(x)
        x = hn[-1]
        x = self.fc(x)
        return x

# Instantiate the CNN-RNN model
bert_cnn_rnn_model = BertCNNRNN(output_size=1)  # Output size is 1 for sentiment score

# Function to prepare inputs for the model
def prepare_input(comment):
    inputs = bert_tokenizer(comment, return_tensors='pt', padding=True, truncation=True, max_length=512)
    return inputs['input_ids']

def categorize_sentiment(score):
    if score >= 0.8:
        return 'Very Positive'
    elif 0.6 <= score < 0.8:
        return 'Positive'
    elif 0.2 <= score < 0.6:
        return 'Neutral'
    elif -0.2 <= score < 0.2:
        return 'Negative'
    elif score < -0.2:
        return 'Very Negative'

def analyze_sentiment(comment):
    """Analyze the sentiment of the comment using VADER, RoBERTa, and BERT-CNN-RNN models."""
    
    # VADER score
    vader_score = vader_analyzer.polarity_scores(comment)['compound']
    
    # RoBERTa score
    roberta_score = roberta_analyzer(comment)[0]['score']
    
    # BERT-CNN-RNN score
    bert_input = prepare_input(comment)
    with torch.no_grad():
        bert_output = bert_cnn_rnn_model(bert_input)
    bert_score = torch.sigmoid(bert_output).item()  # Sigmoid to convert to probability (0-1)

    # Average the scores
    avg_score = (vader_score + roberta_score + bert_score) / 3
    sentiment_score=avg_score

    # Categorize sentiment based on the average score
    sentiment_category = categorize_sentiment(avg_score)
    
    return sentiment_category, sentiment_score

# youtube_comment_fetcher.py (Update generate_pie_chart and generate_wordcloud)

def generate_pie_chart(sentiments):
    labels = list(sentiments.keys())
    sizes = list(sentiments.values())
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.savefig('static/pie_chart.png')  # Save the pie chart as an image
    plt.close()

            



def save_results_to_csv(video_id, comments, sentiments, sentiment_values, likes):
    """Save the sentiment analysis results to a CSV file."""
    with open(f'{video_id}_sentiment_analysis_results.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Comment', 'Sentiment', 'Sentiment Value', 'Likes'])
        for comment, sentiment, sentiment_value, like_count in zip(comments, sentiments, sentiment_values, likes):
            writer.writerow([comment, sentiment, sentiment_value, like_count])
    print(f"Sentiment analysis results saved to '{video_id}_sentiment_analysis_results.csv'")

def display_comment_statistics(total_comments, sentiment_counts):
    """Display sentiment statistics."""
    print(f"\n--- Comment Statistics ---")
    print(f"Total Comments: {total_comments}")
    print(f"Very Positive Comments: {sentiment_counts['Very Positive']}")
    print(f"Positive Comments: {sentiment_counts['Positive']}")
    print(f"Neutral Comments: {sentiment_counts['Neutral']}")
    print(f"Negative Comments: {sentiment_counts['Negative']}")
    print(f"Very Negative Comments: {sentiment_counts['Very Negative']}")
    print(f"-----------------------------------")
