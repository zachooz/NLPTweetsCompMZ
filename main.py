import tensorflow as tf
from model import TweetClassifier
import preprocess
from tqdm import tqdm
import numpy as np
import pandas as pd

def train(dataset, model):
    epochs = 10
    # shuffled = dataset.shuffle(1000, reshuffle_each_iteration=True)
    batched = dataset.batch(model.batchSize)

    for epoch in range(epochs):
        losses = []
        for step, (tweetids, keywords, locations, texts, masks, lengths, targets) in enumerate(tqdm(batched)):


            with tf.GradientTape() as tape:
                classPredictions, nextWordPredictions = model(texts, masks)
                # zippedPredictions = zip(classPredictions, nextWordPredictions)
                # batchloss = 0
                # for 
                
                classLoss = model.classLossFunc(tf.one_hot(targets, 2), classPredictions)
                nopadTexts = texts[:, 1:lengths[0]]

                nopadPredictions = nextWordPredictions[:, :lengths[0]-1, :]
                onehotNextWords = tf.one_hot(nopadTexts, 30522)
                
                nextWordLoss = model.nextWordLossFunc(onehotNextWords, nopadPredictions)

                summedLoss = nextWordLoss + classLoss
                losses.append(summedLoss)

                gradients = tape.gradient(summedLoss, model.trainable_variables)

                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print("epoch loss", np.exp(np.mean(losses)))
    
    model.save_weights('./data/weights.tmd')

def test(dataset, model):
    batched = dataset.batch(model.batchSize)
    losses = []
    correct = 0
    total = 0
    for step, (tweetids, keywords, locations, texts, masks, lengths, targets) in enumerate(tqdm(batched)):
        classPredictions, nextWordPredictions = model(texts, masks)
                
        classLoss = model.classLossFunc(tf.one_hot(targets, 2), classPredictions)
        nopadTexts = texts[:, 1:lengths[0]]

        nopadPredictions = nextWordPredictions[:, :lengths[0]-1, :]
        onehotNextWords = tf.one_hot(nopadTexts, 30522)
                
        nextWordLoss = model.nextWordLossFunc(onehotNextWords, nopadPredictions)

        summedLoss = nextWordLoss + classLoss
        losses.append(summedLoss)
        
        if(tf.argmax(classPredictions[0]).numpy() == targets[0].numpy()):
            correct += 1
        total += 1

    print("epoch loss", np.exp(np.mean(losses)))
    print("accuracy", correct/total)
    
def predictAndWrite(dataset, model):
    batched = dataset.batch(model.batchSize)
    ids = []
    predictions = []

    for step, (tweetids, keywords, locations, texts, masks) in enumerate(tqdm(batched)):
        classPredictions, nextWordPredictions = model(texts, masks)
        for tweetid in tweetids.numpy():
            ids.append(tweetid)
        for pred in classPredictions.numpy():
            predictions.append(np.argmax(pred))
            
    df = pd.DataFrame(data={
        'id': ids,
        'target': predictions
    })
    df.to_csv('output.csv', index=False)

if __name__ == "__main__":
    model = TweetClassifier()

    full_dataset = preprocess.preprocess("train.csv")
    dataset = full_dataset.shard(num_shards=2, index=0)
    testDataset = full_dataset.shard(num_shards=2, index=1)
    evalDataset = preprocess.preprocess("test.csv", train=False)

    train(dataset, model)
    test(testDataset, model)
    predictAndWrite(evalDataset, model)