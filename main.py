import tensorflow as tf
from model import TweetClassifier
import preprocess


def train(dataset, model):
    epochs = 5
    shuffled = dataset.shuffle(1000, reshuffle_each_iteration=True)
    batched = shuffled.batch(model.batchSize)
    for epoch in range(epochs):
        losses = []
        for step, (tweetids, keywords, locations, texts, masks, targets) in enumerate(batched):
            
            with tf.GradientTape() as tape:
                predictions = model.call(texts, masks)
                loss = model.lossFunc(targets, predictions)
                losses.append(loss)
                gradients = tape.gradient(loss, model.OutputLayer.trainable_weights)
                model.optimizer.apply_gradients(zip(gradients, model.OutputLayer.trainable_weights))
        
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