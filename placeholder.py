import pandas as pd
import numpy as np
import re
import nltk
from nltk import sent_tokenize
# nltk.download('punkt')

pd = pd.read_csv('data/connectwise/pon.csv')

# Sample text
sample_text = pd['Detail Description'][0]
 
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
 
# Tokenize the text into sentences using NLTK
sentences = sent_tokenize(sample_text)
 
# Extract dates and emails from each sentence
all_dates = []
all_emails = []
 
for sentence in sentences:
    all_dates.extend(extract_dates(sentence))
    all_emails.extend(extract_emails(sentence))
 
# Replace dates and emails with placeholders
text_with_placeholders = sample_text
text_with_placeholders = replace_entities(text_with_placeholders, all_dates, "DATE")
text_with_placeholders = replace_entities(text_with_placeholders, all_emails, "EMAIL")
 
# Print the results
print("Original Text:", sample_text)
print("\nText with Placeholders:", text_with_placeholders)


