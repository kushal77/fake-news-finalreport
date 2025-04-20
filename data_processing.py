import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter
import csv
import ast
import contractions
import matplotlib.pyplot as plt

csv.field_size_limit(100000000)

nltk.download('stopwords')

def tokenizer(text):
    # contractions will remove any additional english contractions such as "'s", "n't"
    # this only works in english chars, verifying if there is non english chars
    if text.isascii():
        text = contractions.fix(text)
    tokens = nltk.word_tokenize(text)
    return tokens

# features to check from data
url_pattern = re.compile(r'https?://\S+|www\.\S+')
# should match formats such as "%m %d,%Y", "%m %d" and numeric dates in format YYYY-MM-DD, DD/MM/YYYY, MM-DD-YYY
date_pattern = re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:, \d{4})?\b | \b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b | \b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b')
# should match decimals and integers
numbers_pattern = re.compile(r'\b\d+(\.\d+)?\b')
url_count = 0
date_count = 0
numbers_count = 0

def count_features(text):
    urls = len(url_pattern.findall(text))
    dates = len(date_pattern.findall(text))
    nums = len(numbers_pattern.findall(text))
    return urls, dates, nums


train_data_path = 'dataset/train_data.csv'
tokens_output_file_path = 'dataset/train_data_tokens.csv'

chunks = []
with open(tokens_output_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["tokens"])
    for chunk in pd.read_csv(train_data_path , chunksize=10000, on_bad_lines='warn', engine='python'):
        # remove NaN values from training data to apply tokenization
        chunk['content'] = chunk['content'].astype(str).fillna('')
        chunk_tokens = (chunk['content'].apply(tokenizer))
        writer.writerows([tokens] for tokens in chunk_tokens)
        chunks.append(chunk)
        for content in chunk['content']:
            urls, dates, nums = count_features(content)
            url_count += urls
            date_count += dates
            numbers_count += nums

# train_data
train_data  = pd.concat(chunks, ignore_index=True)


# check how divided the data is
print(train_data['type'].value_counts())
# show features in text
print(f"Total URLs: {url_count}")
print(f"Total Dates: {date_count}")
print(f"Total Numbers: {numbers_count}")

# get most common tokens from the tokenized file
stop_words = stopwords.words('english')
punct = [char for char in string.punctuation]
# remove additional punctuation chars
extra_punct = ['’', '“', '”', '—', '–', '‘', '...', '…', '``', "''", '--']

token_counter = Counter()
stop_words_token_counter = Counter()
stemmed_token_counter = Counter()

stemmer = SnowballStemmer('english')

with open(tokens_output_file_path, newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader) # skipping header 'token'
    for row in reader:
        # the format of each row is string and needs to be converted to list
        tokens = ast.literal_eval(row[0])
        tokens = [token for token in tokens if token not in punct and token not in extra_punct]
        token_counter.update(tokens)
        # remove stopwords and add them to separate counter
        tokens_without_stopwords = [token for token in tokens if token.lower() not in stop_words]
        stop_words_token_counter.update(tokens_without_stopwords)
        # apply stemming
        stemmed_token_counter.update(stemmer.stem(token) for token in tokens_without_stopwords)

most_common_tokens = token_counter.most_common(100)
print(f"100 most common tokens: {most_common_tokens}")
most_common_tokens_without_stopwords = stop_words_token_counter.most_common(100)
print(f"100 most common tokens without stopwords: {most_common_tokens_without_stopwords}")
most_common_tokens_after_stemming = stemmed_token_counter.most_common(100)
print(f"100 most common tokens after stemming: {most_common_tokens_after_stemming}")

original_vocabulary_size = token_counter.total()
stop_words_vocabulary_size = stop_words_token_counter.total()
stemmed_vocabulary_size = stemmed_token_counter.total()

# Compute reduction rates
stopwords_reduction_rate = ((original_vocabulary_size- stop_words_vocabulary_size) / original_vocabulary_size) * 100
stemming_reduction_rate = ((stop_words_vocabulary_size - stemmed_vocabulary_size) / stop_words_vocabulary_size) * 100
print(f"Original vocabulary size: {original_vocabulary_size}")
print(f"Vocabulary size after removing stopwords: {stop_words_vocabulary_size}")
print(f"Reduction rate after removing stopwords: {stopwords_reduction_rate}")
print(f"Vocabulary size after stemming: {stemmed_vocabulary_size}")
print(f"Reduction rate after stemming: {stemming_reduction_rate}")

# visualize most common words
words, counts = zip(*most_common_tokens_without_stopwords)
plt.figure(figsize=(15, 5))
plt.bar(words, counts)
plt.xticks(rotation=90)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 100 Most Frequent Words After Stopwords")
plt.show()
