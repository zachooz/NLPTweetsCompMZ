import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

class TweetClassifier(tf.keras.Model):
    def __init__(self):
        super(TweetClassifier, self).__init__()

        self.Optimizer = tf.keras.optimizers.Adam()

        self.lossFunc = tf.keras.losses.BinaryCrossentropy()

        self.batchSize = 10

        self.BertModel = TFBertModel.from_pretrained('bert-base-uncased')

        self.OutputLayer = tf.keras.layers.Dense(1)

    
    def call(self, inputs, masks):
        bertOutputs = self.BertModel(inputs, attention_mask=masks)
        pooler_output = bertOutputs[1]
        finalOutput = self.OutputLayer(pooler_output)
        
        return finalOutput