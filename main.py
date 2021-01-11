import tensorflow as tf
from model import TweetClassifier, BertModel
import preprocess
from tqdm import tqdm
import numpy as np
import pandas as pd

def train(dataset, model):
    epochs = 1
    shuffled = dataset.shuffle(1000, reshuffle_each_iteration=True)
    batched = shuffled.batch(model.batchSize)
    bmodel = BertModel()

    for epoch in range(epochs):
        losses = []
        for step, (tweetids, keywords, locations, texts, masks, targets) in enumerate(tqdm(batched)):

            poolerOutputs = bmodel.pool(texts, masks)

            with tf.GradientTape() as tape:
                predictions = model(poolerOutputs)
                loss = model.lossFunc(targets, predictions)
                losses.append(loss)

                gradients = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print("epoch loss", np.exp(np.mean(losses)))
    
    model.save_weights('./data/weights.tmd')


def predictAndWrite(dataset, model, outputFile):
    batched = dataset.batch(model.batchSize)
    ids = []
    predictions = []
    for step, (tweetids, keywords, locations, texts, masks) in enumerate(batched):
        for id in tweetids:
            ids.append(id)
        for pred in predictions:
            if pred > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

    df = pd.DataFrame(data={
        'id': ids,
        'target': predictions
    })
    df.to_csv('output.csv', index=Fas)

def main():
    dataset = preprocess.preprocess("train.csv")
    model = TweetClassifier()
    
    train(dataset, model)

    outputFile = open("output.txt", "w+")

    evalDataset = preprocess.preprocess("test.csv", train=False)

    predictAndWrite(evalDataset, model, outputFile)


if __name__ == "__main__":
    main()
