# Import necessary libraries
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Sample text data
text_data = "This is a sample text with numbers like 123, dates such as 01/20/2022, and special characters !@#$%. Let's normalize it."

# Tokenization
tokens = word_tokenize(text_data)
print("Original Text:", tokens)

# Stop Word Removal
stop_words = set(stopwords.words("english"))
filtered_words = [word for word in tokens if word.lower() not in stop_words]
print("Stop Word Removal:", filtered_words)

# Normalization
normalized_text = []

for word in filtered_words:
    # Lowercasing
    word = word.lower()

    # Convert numbers to a standard format
    if re.match(r'\d+', word):
        word = 'NUM'

    # Convert dates to a standard format (assuming in MM/DD/YYYY format)
    if re.match(r'\b(?:\d{1,2}/){2}\d{4}\b', word):
        word = 'DATE'

    # Remove special characters
    word = re.sub(r'[^\w\s]', '', word)

    # Add to normalized text
    normalized_text.append(word)

print("Normalization:", normalized_text)