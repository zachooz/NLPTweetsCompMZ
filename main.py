import tensorflow as tf
from model import TweetClassifier
import preprocess
from tqdm import tqdm
import numpy as np
import pandas as pd

def train(dataset, model):
    epochs = 1
    shuffled = dataset.shuffle(1000, reshuffle_each_iteration=True)
    batched = shuffled.batch(model.batchSize)

    for epoch in range(epochs):
        losses = []
        for step, (tweetids, keywords, locations, texts, masks, targets) in enumerate(tqdm(batched)):

            with tf.GradientTape() as tape:
                predictions = model(texts)
                loss = model.lossFunc(targets, predictions)
                losses.append(loss)

                gradients = tape.gradient(loss, model.outputLayer.trainable_variables)
                model.optimizer.apply_gradients(zip(gradients, model.outputLayer.trainable_variables))

        print("epoch loss", np.exp(np.mean(losses)))

    model.save_weights('./data/weights.tmd')



def predictAndWrite(dataset, model):
    batched = dataset.batch(model.batchSize)
    ids = []
    predictions = []

    for step, (tweetids, keywords, locations, texts, masks) in enumerate(tqdm(batched)):
        classPredictions = model(texts)
        for tweetid in tweetids.numpy():
            ids.append(tweetid)
        for pred in classPredictions.numpy():
            if pred > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
            
    df = pd.DataFrame(data={
        'id': ids,
        'target': predictions
    })
    df.to_csv('output.csv', index=False)

def test(dataset, model):
    batched = dataset.batch(model.batchSize)
    losses = []
    correct = 0
    total = 0
    for step, (tweetids, keywords, locations, texts, masks, lengths, targets) in enumerate(tqdm(batched)):
        classPrediction = model(texts, masks)
                
        classLoss = model.lossFunc(tf.one_hot(targets, 2), classPrediction)
        losses.append(classLoss)
        
        if(tf.argmax(classPredictions[0]).numpy() == targets[0].numpy()):
            correct += 1
        total += 1

    print("epoch loss", np.exp(np.mean(losses)))
    print("accuracy", correct/total)
    

def main():
    dataset = preprocess.preprocess("train.csv")
    model = TweetClassifier()

    evalDataset = preprocess.preprocess("test.csv", train=False)

    train(dataset, model)



    predictAndWrite(evalDataset, model)


if __name__ == "__main__":
    main()
