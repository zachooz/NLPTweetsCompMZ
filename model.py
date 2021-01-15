import tensorflow as tf
from transformers import BertTokenizer, TFBertForPreTraining



class TweetClassifier(tf.keras.Model):
    def __init__(self):
        super(TweetClassifier, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.classLossFunc = tf.keras.losses.CategoricalCrossentropy()
        self.nextWordLossFunc = tf.keras.losses.CategoricalCrossentropy()

        self.bert = TFBertForPreTraining.from_pretrained('bert-base-uncased')
        self.batchSize = 1

    def call(self, inputs, masks):
        bertOutput = self.bert(inputs, attention_mask=masks, return_dict=True)

        # next word predictions (batch_size, sequence_length, config.vocab_size)
        logits = bertOutput.prediction_logits 

        # hidden states (batch_size, sequence_length, hidden_size)
        # hidden_states = bertOutput.hidden_states
        
        # # wish to get the hidden state of the first element [CLS] of each sequence 
        # dl1Output = self.Dl1(hidden_states[:, 0, :])
        # finalOutput = self.OutputLayer(dl1Output)
        
        # (batch_size, 2), (batch_size, sequence_length, config.vocab_size)
        return bertOutput.seq_relationship_logits, logits

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
