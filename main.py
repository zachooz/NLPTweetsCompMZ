import tensorflow as tf
from model import TweetClassifier, BertModel
import preprocess
from tqdm import tqdm
import numpy as np
import pandas as pd

def train(dataset, model, bm):
    epochs = 1
    shuffled = dataset.shuffle(1000, reshuffle_each_iteration=True)
    batched = shuffled.batch(model.batchSize)

    for epoch in range(epochs):
        losses = []
        batched = tqdm(batched)
        for step, (tweetids, keywords, locations, texts, masks, targets) in enumerate(batched):
            poolerOut = bm.call(texts)
    
            with tf.GradientTape() as tape:
                predictions = model(poolerOut)
                loss = model.lossFunc(tf.one_hot(targets, 2), predictions)
                lossnp = np.mean(loss.numpy())
                losses.append(lossnp)
                batched.set_postfix({'loss': lossnp})

                gradients = tape.gradient(loss, model.outputLayer.trainable_variables)
                model.optimizer.apply_gradients(zip(gradients, model.outputLayer.trainable_variables))

        print("epoch loss", np.mean(losses))

    model.save_weights('./data/weights.tmd')



def predictAndWrite(dataset, model, bm):
    batched = dataset.batch(model.batchSize)
    ids = []
    predictions = []

    for step, (tweetids, keywords, locations, texts, masks) in enumerate(tqdm(batched)):
        poolerOut = bm.call(texts)
        classPredictions = model(poolerOut)
        for tweetid in tweetids.numpy():
            ids.append(tweetid)
        for pred in classPredictions.numpy():
            predictions.append(np.argmax(pred))
            
    df = pd.DataFrame(data={
        'id': ids,
        'target': predictions
    })
    df.to_csv('output.csv', index=False)

def test(dataset, model, bm):
    batched = dataset.batch(model.batchSize)
    losses = []
    correct = 0
    total = 0
    for step, (tweetids, keywords, locations, texts, masks, targets) in enumerate(tqdm(batched)):
        poolerOut = bm.call(texts)
        classPrediction = model(poolerOut)
                
        classLoss = model.lossFunc(tf.one_hot(targets, 2), classPrediction)
        classPrediction = classPrediction.numpy()
        losses.append(np.mean(classLoss.numpy()))
        for i in range(classPrediction.shape[0]):
            if(np.argmax(classPrediction[i]) == targets.numpy()[i]):
                correct += 1
        total += 1

    print("epoch loss", np.mean(losses))
    print("accuracy", correct/total)
    

def main():
    bm = BertModel()
    dataset = preprocess.preprocess("train.csv")
    model = TweetClassifier()

    full_dataset = preprocess.preprocess("train.csv")
    dataset = full_dataset.shard(num_shards=2, index=0)
    testDataset = full_dataset.shard(num_shards=2, index=1)
    evalDataset = preprocess.preprocess("test.csv", train=False)

    train(dataset, model, bm)
    test(testDataset, model, bm)

    predictAndWrite(evalDataset, model, bm)


if __name__ == "__main__":
    main()
