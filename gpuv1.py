#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from dataclasses import dataclass
import numpy as np
import glob
import re
from pprint import pprint

from nltk import pos_tag,ngrams
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle

# In[ ]:


@dataclass
class Config:
    MAX_LEN = 256
    BATCH_SIZE = 32
    LR = 0.001
    VOCAB_SIZE = 30000
    EMBED_DIM = 128
    NUM_HEAD = 8  # used in bert model
    FF_DIM = 128  # used in bert model
    NUM_LAYERS = 3


config = Config()


# # Data

# In[ ]:


with open("data/wiki_en.txt",encoding="utf-8",errors="replace") as f:
    data = f.readlines()
    
print("Data loaded into memory!")


# In[ ]:


length_checker = np.vectorize(len)
detokenizer = TreebankWordDetokenizer().detokenize
def replacer(line,max_=2):
    max_ = np.random.randint(2,max_+1)
    t = np.array(wordpunct_tokenize(line))
    orig = t.copy()
    arr_len = length_checker(t)
    grams = np.where(arr_len>1)[0]
    try:
        indices = np.array(grams[np.random.choice(len(grams),max_,replace=False)])
    except:
        return detokenizer(orig),detokenizer(t)
    t_indices = indices.copy()
    np.random.shuffle(t_indices)
    while (t_indices == indices).any():
        np.random.shuffle(t_indices)
    t[indices] = t[t_indices]
    return detokenizer(orig),detokenizer(t)


# In[ ]:


X = []
y = []
num_samples = 3
for i in tqdm(data):
    for j in range(2,num_samples+2):
        x_,y_ = replacer(i,j)
        X.append(x_)
        y.append(y_)
print(len(X),len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
try:
    with open("data/processed_data.pkl","wb") as f:
        pickle.dump({"X_train":X_train,"y_train":y_train,"X_test":X_test,"y_test":y_test}, f, protocol=pickle.HIGHEST_PROTOCOL)
except:
    pass
print("Data jumbled,splitted and saved!")


# In[ ]:


vectorizer = tf.keras.layers.Texttrain_datatorization(
    max_tokens=config.VOCAB_SIZE, standardize='lower_and_strip_punctuation',
    split='whitespace', ngrams=None, output_mode='int',
    output_sequence_length=config.MAX_LEN, pad_to_max_tokens=True,
)
vectorizer.adapt(X_train)
vocab = vectorizer.get_vocabulary()

id2token = dict(enumerate(vocab))
token2id = {y: x for x, y in id2token.items()}
print("Vectorizer Initialized!")
# Create model.
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
model.add(vectorizer)

# Save.
filepath = "data/vectorizer"
model.save(filepath, save_format="tf")

# Load.
# loaded_model = tf.keras.models.load_model(filepath)
# loaded_vectorizer = loaded_model.layers[0]
print("Vectorizer saved!")


# In[ ]:


def get_data(X_train,y_train):
    def encode(texts):
        encoded_texts = vectorizer(texts)
        return encoded_texts.numpy()
    X_train_ = encode(X_train)
    y_train_ = encode(y_train)
    input_mask = np.ones(labels.shape)
    input_mask[X_train_ == 0] = 0
    return tf.data.Dataset.from_tensor_slices((X_train_, y_train_, input_mask))

train_ds = get_data(X_train,y_train)
test_ds = get_data(X_test,y_test)

train_ds = train_ds.batch(32)
test_ds = test_ds.batch(32)
print("Data Ready!")


# # Model

# In[ ]:


def bert_module(query, key, value, i):
    # Multi headed self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=config.NUM_HEAD,
        key_dim=config.EMBED_DIM // config.NUM_HEAD,
        name="encoder_{}/multiheadattention".format(i),
    )(query, key, value)
    attention_output = layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(
        attention_output
    )
    attention_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
    )(query + attention_output)

    # Feed-forward layer
    ffn = keras.Sequential(
        [
            layers.Dense(config.FF_DIM, activation="relu"),
            layers.Dense(config.EMBED_DIM),
        ],
        name="encoder_{}/ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1, name="encoder_{}/ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
    )(attention_output + ffn_output)
    return sequence_output

def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


# In[ ]:


cat_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

def loss_fn(y_true, y_pred):
    def size_loss(y_pred_):
        y_true_length = tf.math.count_nonzero(y_true,axis=1)
        y_pred_length = tf.math.count_nonzero(y_pred_,axis=1)
        return tf.reduce_sum(y_true_length-y_pred_length)

    def existing_word_loss(y_pred_):
        return tf.size(y_true) - tf.reduce_sum(tf.cast(tf.equal(y_true,y_pred_), tf.int32))

    def word_position_loss():
        return cat_loss(y_true, y_pred).numpy()
    
    y_pred_ = tf.math.argmax(y_pred,axis=1)
    wp_loss = tf.square(word_position_loss())
    ew_loss = tf.square(tf.cast(existing_word_loss(y_pred_),dtype=tf.float32)) 
    ls_loss = tf.square(tf.cast(size_loss(y_pred_),dtype=tf.float32))
    return tf.sqrt(tf.reduce_mean([wp_loss,ew_loss,ls_loss]))

loss_tracker = tf.keras.metrics.Mean(name="loss")

class LanguageModel(tf.keras.Model):
    def train_step(self, inputs):
        if len(inputs) == 3:
            features, labels, sample_weight = inputs
        else:
            features, labels = inputs
            sample_weight = None

        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            loss = loss_fn(labels, predictions, sample_weight=sample_weight)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result()}
    
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker]
    
def create_jumbled_bert_model():
    inputs = layers.Input((config.MAX_LEN,), dtype=tf.int64)

    word_embeddings = layers.Embedding(
        config.VOCAB_SIZE, config.EMBED_DIM, name="word_embedding"
    )(inputs)
    position_embeddings = layers.Embedding(
        input_dim=config.MAX_LEN,
        output_dim=config.EMBED_DIM,
        weights=[get_pos_encoding_matrix(config.MAX_LEN, config.EMBED_DIM)],
        name="position_embedding",
    )(tf.range(start=0, limit=config.MAX_LEN, delta=1))
    embeddings = word_embeddings + position_embeddings

    encoder_output = embeddings
    for i in range(config.NUM_LAYERS):
        encoder_output = bert_module(encoder_output, encoder_output, encoder_output, i)

    lm_output = layers.Dense(config.VOCAB_SIZE, name="lm_cls", activation="softmax")(
        encoder_output
    )
    lm_model = LanguageModel(inputs, lm_output, name="jumble_bert")

    optimizer = keras.optimizers.Adam(learning_rate=config.LR)
    lm_model.compile(optimizer=optimizer)
    return lm_model

j_bert_model = create_jumbled_bert_model()
print("Model ready!")
print(j_bert_model.summary())


# In[ ]:


checkpoint_filepath = 'data/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)


# In[ ]:


print("Starting training!")
history = j_bert_model.fit(train_ds,epochs = 10, validation_data=test_ds,callbacks=[model_checkpoint_callback])
print("Training Ended!")

