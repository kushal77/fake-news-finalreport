import pandas as pd
from sklearn.model_selection import train_test_split
import csv
file_path = "dataset/news_cleaned_2018_02_13.csv"
csv.field_size_limit(100000000)


selected_columns = ["id", "domain", "type", "url", "content", "title"]
sample_chunks = []
for chunk in pd.read_csv(file_path, usecols=selected_columns, chunksize=100000, on_bad_lines='warn', engine='python'):
    sample = chunk.sample(frac=0.1, random_state=42)
    sample_chunks.append(sample)
    df = pd.concat(sample_chunks, ignore_index=True)

train_dataframe, test_dataframe = train_test_split(df, test_size=0.2, random_state=42)


train_dataframe.to_csv("dataset/train_data.csv", chunksize=100000)
test_dataframe.to_csv("dataset/test_data.csv", chunksize=100000)
