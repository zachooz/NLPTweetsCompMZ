import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizerFast

def preprocess(path, train=True):
    bertTokenizer = BertTokenizerFast.from_pretrained('bert-large-cased')

    df = pd.read_csv(path).replace(np.nan, '', regex=True).to_numpy()
    
    data_dict = {
        "id": 0,
        "keyword": 1,
        "location": 2,
        "text": 3,
        "target": 4,
    }

    tweetids = []
    keywords = []
    locations = []
    texts = []
    targets = []
    masks = []
    lengths = []

    for row in df:
        tweetid = row[data_dict["id"]]
        keyword = row[data_dict["keyword"]]
        location = row[data_dict["location"]]
        text = row[data_dict["text"]]
        

        tweetids.append(tweetid)
        keywords.append(bertTokenizer.convert_tokens_to_ids(keyword) if keyword != '' else bertTokenizer.convert_tokens_to_ids('unk'))
        
        locations.append(bertTokenizer.convert_tokens_to_ids(location) if location != '' else bertTokenizer.convert_tokens_to_ids('unk'))
        tokenized = bertTokenizer.encode(text)
        texts.append(tokenized)
        lengths.append(len(tokenized))
        masks.append([1] *  len(tokenized))

        if train:
            target = row[data_dict["target"]]
            targets.append(target)

    pad_token = bertTokenizer.pad_token
    pad_token = bertTokenizer.convert_tokens_to_ids(pad_token)
    texts = tf.keras.preprocessing.sequence.pad_sequences(texts, padding='post', value=pad_token)
    masks = tf.keras.preprocessing.sequence.pad_sequences(masks, padding='post', value=0)
    if train:
        dataset = tf.data.Dataset.from_tensor_slices((tweetids, keywords, locations, texts, masks, lengths, targets))
        return dataset

    dataset = tf.data.Dataset.from_tensor_slices((tweetids, keywords, locations, texts, masks))
    return dataset

def example_enumerate():
    dataset = preprocess('train.csv')

    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    dataset = dataset.batch(2)

    for step, (tweetids, keywords, locations, texts, masks, lengths, targets) in enumerate(dataset):
        print(tweetids, keywords, locations, texts, masks, lengths, targets)
        break

if __name__ == "__main__":
    # bertTokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # sent = 'bob walked across the street.'
    # e = bertTokenizer.encode(sent)
    # d = bertTokenizer.decode(e)
    # print(sent)
    # print(e)
    # print(d)

    # print(bertTokenizer.encode([
    #     "Hi I am bobby",
    #     "wow what"
    # ]))
    example_enumerate()
