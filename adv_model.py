import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import tensorflow_text as text
from keras.metrics import Recall
from tensorflow import keras
import csv
from tensorflow.python.keras.metrics import Precision, Recall

csv.field_size_limit(100000000)
selected_columns = ["type", "content"]
# get dataset
train_file_path = "dataset/train_data.csv"
train_data_chunks = []
for chunk in pd.read_csv(train_file_path, nrows=1000, usecols=selected_columns, chunksize=100000, on_bad_lines='warn', engine='python'):
    chunk = chunk.dropna()
    train_data_chunks.append(chunk)

train_data = pd.concat(train_data_chunks, ignore_index=True)

test_file_path = "dataset/test_data.csv"
test_data_chunks = []
for chunk in pd.read_csv(test_file_path, nrows=1000, usecols=selected_columns, chunksize=100000, on_bad_lines='warn', engine='python'):
    chunk = chunk.dropna()
    test_data_chunks.append(chunk)

test_data = pd.concat(test_data_chunks, ignore_index=True)

# encode labels (=types) for classification

# type in dataset: fake, satire, bias, conspiracy, state, junksci, hate, clickbait, unreliable, political, reliable
# Encode labels consistently
#type_categories = sorted(train_data['type'].unique())  # Ensure consistent label mapping
#train_labels = train_data['type'].astype(pd.CategoricalDtype(categories=type_categories)).cat.codes
#test_labels = test_data['type'].astype(pd.CategoricalDtype(categories=type_categories)).cat.codes

#num_classes = len(type_categories)
#train_y = keras.utils.to_categorical(train_labels, num_classes)
#test_y = keras.utils.to_categorical(test_labels, num_classes)
#

# test only with 2 labels fake and not fake
fake_labels = {'fake', 'satire', 'bias', 'conspiracy', 'state', 'junksci', 'hate', 'clickbait', 'unreliable'}
not_fake_labels = {'political', 'reliable'}
train_data['binary_label'] = train_data['type'].apply(lambda x: 1 if x in not_fake_labels else 0)
test_data['binary_label'] = test_data['type'].apply(lambda x: 1 if x in not_fake_labels else 0)
train_y = np.array(train_data['binary_label'])
test_y = np.array(test_data['binary_label'])

# bert model
bert_model_name = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
preprocessor_name = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

bert_preprocess = hub.KerasLayer(preprocessor_name)
bert_encoder = hub.KerasLayer(bert_model_name, trainable=False)

keras.mixed_precision.set_global_policy('mixed_float16')
def build_model():
    text_input = keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessing_layer = bert_preprocess(text_input)
    encoder_outputs = bert_encoder(preprocessing_layer)["pooled_output"]
    dropout = keras.layers.Dropout(0.3)(encoder_outputs)
    #output = keras.layers.Dense(num_classes, activation="softmax")(dropout)
    output = keras.layers.Dense(1, activation="sigmoid")(dropout)
    model = keras.Model(inputs=text_input, outputs=output)
    #model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy', Precision(), Recall()])
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy', Precision(), Recall()])
    return  model

model = build_model()

history = model.fit(train_data['content'], train_y, epochs=3, batch_size=32, validation_data=(test_data['content'], test_y))
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_data['content'], test_y)
f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)

print(f"Results: \n Test accuracy: {test_accuracy} \n Test loss: {test_loss} \n Test precision: {test_precision} \n Test recall: {test_recall} \n Test F1_Score: {f1_score}")
