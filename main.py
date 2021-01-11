import tensorflow as tf
from model import TweetClassifier
import preprocess
from tqdm import tqdm
import numpy as np
import pandas as pd

def train(dataset, model):
    epochs = 50
    shuffled = dataset.shuffle(1000, reshuffle_each_iteration=True)
    batched = shuffled.batch(model.batchSize)

    for epoch in range(epochs):
        losses = []
        for step, (tweetids, keywords, locations, texts, masks, targets) in enumerate(tqdm(batched)):


            with tf.GradientTape() as tape:
                classPredictions, nextWordPredictions = model(texts, masks)
                classLoss = model.classLossFunc(tf.one_hot(targets, 2), classPredictions)
                onehotNextWords = tf.one_hot(texts[1:], 30522)
                
                nextWordLoss = model.nextWordLossFunc(onehotNextWords, nextWordPredictions)

                summedLoss = nextWordLoss + classLoss
                losses.append(summedLoss)

                gradients = tape.gradient(summedLoss, model.trainable_variables)

                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print("epoch loss", np.exp(np.mean(losses)))
    
    model.save_weights('./data/weights.tmd')

# def test(dataset, model, bmodel):
#     losses = []
#     batched = dataset.batch(model.batchSize)
#     for step, (tweetids, keywords, locations, texts, masks, targets) in enumerate(tqdm(batched)):

#         poolerOutputs = bmodel.pool(texts, masks)
#         predictions = model(poolerOutputs)
#         loss = model.lossFunc(targets, predictions)
#         losses.append(loss)

#     print("epoch loss", np.exp(np.mean(losses)))

# def predictAndWrite(dataset, model, bmodel):
#     batched = dataset.batch(model.batchSize)
#     ids = []
#     predictions = []

#     for step, (tweetids, keywords, locations, texts, masks) in enumerate(batched):
#         poolerOutputs = bmodel.pool(texts, masks)
#         preds = model(poolerOutputs)
#         for tweetid in tweetids.numpy():
#             ids.append(tweetid)
#         for pred in preds.numpy().flatten():
#             if pred > 0.5:
#                 predictions.append(1)
#             else:
#                 predictions.append(0)
#     print(len(predictions), len(ids))
#     df = pd.DataFrame(data={
#         'id': ids,
#         'target': predictions
#     })
#     df.to_csv('output.csv', index=False)

def main():
    full_dataset = preprocess.preprocess("train.csv")
    dataset = full_dataset.shard(num_shards=2, index=0)
    # testDataset = full_dataset.shard(num_shards=2, index=1)
    #evalDataset = preprocess.preprocess("test.csv", train=False)
    model = TweetClassifier()
    train(dataset, model)
    # test(testDataset, model, bmodel)
    # predictAndWrite(evalDataset, model, bmodel)

if __name__ == "__main__":
    main()
