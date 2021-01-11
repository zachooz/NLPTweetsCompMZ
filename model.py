import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

class BertModel:
    def __init__(self):
        self.model = TFBertModel.from_pretrained('bert-base-uncased')

    def pool(self, inputs, masks):
        bertOutputs = self.model(inputs, attention_mask=masks)
        pooler_output = bertOutputs[0]
        return pooler_output

class TweetClassifier(tf.keras.Model):
    def __init__(self):
        super(TweetClassifier, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.lossFunc = tf.keras.losses.BinaryCrossentropy()
        self.preOutPut = tf.keras.layers.Dense(300, activation='relu')
        self.outputLayer = tf.keras.layers.Dense(1)
        self.batchSize = 50


    def call(self, poolerOutput):
        preOutOutput = self.preOutPut(poolerOutput)
        finalOutput = self.outputLayer(preOutOutput)

        return finalOutput

if __name__ == "__main__":
    layer = tf.keras.layers.Dense(1, activation='relu')
    x = tf.constant([[1., 2., 3.]])

    with tf.GradientTape() as tape:
        # Forward pass
        y = layer(x)
        loss = tf.reduce_mean(y**2)

    # Calculate gradients with respect to every trainable variable
    grad = tape.gradient(loss, layer.trainable_variables)
    print(layer.trainable_variables)
