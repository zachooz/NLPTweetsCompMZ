import tensorflow as tf
from transformers import TFAutoModel

class BertModel:
    def __init__(self):
        self.berttweet = TFAutoModel.from_pretrained("vinai/bertweet-base")
    
    def call(self, inputs):
        return self.berttweet(inputs, return_dict = True)['pooler_output']

class TweetClassifier(tf.keras.Model):
    def __init__(self):
        super(TweetClassifier, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=2e-3)
        self.lossFunc = tf.keras.losses.BinaryCrossentropy()
        
        # self.preOutPut = tf.keras.layers.Dense(300, activation='relu')
        self.outputLayer = tf.keras.layers.Dense(2)
        self.batchSize = 8


    def call(self, inputs):
       
        # preOutOutput = self.preOutPut(poolerOutput)
        finalOutput = self.outputLayer(inputs)

        return tf.nn.softmax(finalOutput)

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
