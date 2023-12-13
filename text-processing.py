import pandas as pd
import numpy as np
import re
import nltk
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import calendar
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt')

df = pd.read_csv('data/connectwise/pon.csv')

# Sample text
sample_text = df['Detail Description'][0]


# Function to extract dates using regular expressions
def extract_dates(text):
    date_pattern = r'\b\d{2}/\d{2}/\d{4}\b'
    return re.findall(date_pattern, text)
 
# Function to extract emails using regular expressions
def extract_emails(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)
 
# Function to replace dates and emails with placeholders
def replace_entities(text, entities, placeholder):
    for entity in entities:
        text = text.replace(entity, placeholder)
    return text

def extract_names_from_emails(emails):
    names = []
    name_pattern = r"([A-Za-z]+)\.([A-Za-z]+)@" 

    for email in emails:
        match = re.search(name_pattern, email)
        if match:
            first_name = match.group(1)
            last_name = match.group(2)
            names.append(first_name.lower())
            names.append(last_name.lower())  

    return names
 
def find_top_words(text):

    # Tokenize the text into sentences using NLTK
    sentences = sent_tokenize(text)
    
    # Extract dates and emails from each sentence
    all_dates = set()
    all_emails = set()

    for sentence in sentences:
        dates = extract_dates(sentence)
        emails = extract_emails(sentence)

        # Add dates to the set
        all_dates.update(dates)

        # Add emails to the set
        all_emails.update(emails)

    # Convert sets back to lists if needed
    all_dates = list(all_dates)
    all_emails = list(all_emails)
    
    # Replace dates and emails with placeholders
    text_with_placeholders = text
    text_with_placeholders = replace_entities(text_with_placeholders, all_dates, "DATE")
    text_with_placeholders = replace_entities(text_with_placeholders, all_emails, "EMAIL")

    
    ## Tokenization
    tokens = word_tokenize(text_with_placeholders)
    # print("Original Text:", tokens)

    # Stop Word Removal
    stop_words = set(stopwords.words("english"))

    filtered_words = [word for word in tokens if word.lower() not in stop_words]
    # print("Stop Word Removal:", filtered_words)

    # Normalization
    normalized_text = []

    for word in filtered_words:
        # Lowercasing
        if(word != 'DATE' and word != 'EMAIL'):
            if word != word.upper():
                word = word.lower()

        # Convert numbers to a standard format
        if re.match(r'\d+', word):
            word = 'NUM'
        
         # Remove special characters
        word = re.sub(r'[^\w\s]', '', word)

        if re.match(r'UTC\d+', word):
            word = 'UTC'


        # Add to normalized text
        normalized_text.append(word)

    weekdays = list(calendar.day_name) + list(calendar.month_name)
    weekdays = [day.lower() for day in weekdays]
    names = extract_names_from_emails(all_emails)
    # print("\nNames: ", names)
    filtered_words = [word for word in normalized_text if word not in ['DATE', 'EMAIL', 'NUM', 'UTC', 'please', 'hello', 'regards', 'email', 'quadreal', 'andorix', 'hi', '1711'] and word not in weekdays and word not in names]


    filtered_text = ' '.join(filtered_words)

    # print("Original Text:", text)
    # print("\nText with Placeholders:", text_with_placeholders)
    # print("\Filtered Text:", filtered_words)

    if filtered_text == '':
        return [], []
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)

    # !!!! Plot the word cloud !!!!!!
    
    # plt.figure(figsize=(10, 5))
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')
    # plt.show()

    # # Plot a bar chart (without cumulative plot)
    # plt.figure(figsize=(10, 5))
    # freq_dist.plot(30, cumulative=False)

    # print("\n All Emails: ", all_emails)
    # print("\n All Dates: ", all_dates)

    # Create a frequency distribution for filtered words
    freq_dist = FreqDist(filtered_words)

    top_words_freq = [word for word,_ in freq_dist.most_common(10)]

    # TF-IDF

    # Convert the filtered text into a list of documents
    documents = [filtered_text]

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(lowercase=False)

    # Fit and transform the documents
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Get the TF-IDF values for the first document
    tfidf_values = tfidf_matrix.toarray()[0]

    # Create a dictionary mapping words to their TF-IDF values
    word_tfidf = dict(zip(feature_names, tfidf_values))

    # Sort the dictionary by TF-IDF values in descending order
    sorted_word_tfidf = sorted(word_tfidf.items(), key=lambda x: x[1], reverse=True)

    # Get the top 10 words based on TF-IDF values
    top_words_tf_idf = [word for word, _ in sorted_word_tfidf[:10]]

    print("\nTop Words (Frequency): ", top_words_freq)
    print("Top Words (TF-IDF): ", top_words_tf_idf)

    return top_words_freq, top_words_tf_idf 

df[['top_words_freq', 'top_words_tfidf']] = df['Detail Description'].apply(lambda x: pd.Series(find_top_words(x)))

df.to_csv('connectwise/pon_updated.csv', index=False)