from textblob import TextBlob
def detect_hate_speech(text):
  blob = TextBlob(text)
  sentiment_polarity = blob.sentiment.polarity
  threshold = 0.0
  is_hate_speech = sentiment_polarity < threshold

  return is_hate_speech, sentiment_polarity

text_to_check = "You are a wonderful person"
hate_speech_result, sentiment_polarity_result = detect_hate_speech(text_to_check)

print(f"Is it Hate Speech? {hate_speech_result}")