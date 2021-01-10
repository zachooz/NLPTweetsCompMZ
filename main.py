import tensorflow as tf
from model import TweetClassifier, BertModel
import preprocess
from tqdm import tqdm
import numpy as np

def train(dataset, model):
    epochs = 5
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


def predictAndWrite(dataset, model, outputFile):
    shuffled = dataset.shuffle(1000, reshuffle_each_iteration=True)
    batched = shuffled.batch(model.batchSize)
    numExample = 0
    for step, (tweetids, keywords, locations, texts, masks) in enumerate(batched):
        predictions = tf.math.argmax(model.call(texts, masks))
        for thing in predictions:
            outputFile.write(str(numExample) + "," + str(thing))
            numExample += 1


def main():
    dataset = preprocess.preprocess("train.csv")
    model = TweetClassifier()

    train(dataset, model)
    
    outputFile = open("output.txt", "w+")

    evalDataset = preprocess.preprocess("test.csv", train=False)

    predictAndWrite(evalDataset, model, outputFile)


if __name__ == "__main__":
    main()