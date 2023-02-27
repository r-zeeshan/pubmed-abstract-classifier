from helper_functions import *
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, Concatenate, Dropout, Embedding
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


data_dir = "pubmed-rct-master/PubMed_200k_RCT_numbers_replaced_with_at_sign/"

train_lines = get_lines(data_dir + "train.txt")
train_lines[:3]

"""### We need the data in the following format to be able to use it with our model 

```
[{  'line_number' : 0,
    'target' : 'BACKGROUND',
    'text' : 'Emotional eating is the leading cause of Obesity',
    'total_lines' : 11
}]
```
"""

# Get the data from the file and preprocess it
train_samples = preprocess_text(data_dir + "train.txt")
test_samples = preprocess_text(data_dir + "test.txt")
val_samples = preprocess_text(data_dir + "dev.txt")

print(f"Length of training samples: {len(train_samples)}")
print(f"Length of training samples: {len(test_samples)}")
print(f"Length of training samples: {len(val_samples)}")

# Converting our lists into Pandas dataframes
train_df = pd.DataFrame(train_samples)
test_df = pd.DataFrame(test_samples)
val_df = pd.DataFrame(val_samples)

"""# Getting data ready for our model

Our model will have 4 different inputs 

    1. Line Numbers (One Hot Encoded)
    2. Total Lines (One Hot Encoded)
    3. Train Sentences (Custom Token Embeddings)
    4. Train Chars (Custom character embeddings)

And output will be target Label which we will also encode
"""

# Encoding labels OneHot and LabelEncode
one_hot_encoder = OneHotEncoder(sparse=False)
label_encoder = LabelEncoder()

train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1,1))
test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1,1))
val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1,1))

train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
test_labels_encoded = label_encoder.transform(test_df["target"].to_numpy())
val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())

print(f"\nTrain Labels One Hot Encoded: {train_labels_one_hot}\n")
print(f"\nTrain Labels Encoded: {train_labels_encoded}\n")

num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_

"""## Preparing the text data for our model"""

# Connverting out sentences tolist
train_sentences = train_df["text"].tolist()
test_sentences = test_df["text"].tolist()
val_sentences = val_df["text"].tolist()

# How long is each sentence on average?
sent_lens = [len(sentence.split()) for sentence in train_sentences]
avg_sent_len  = np.mean(sent_lens)
output_seq_len = int(np.percentile(sent_lens, 95))
output_seq_len

"""### Creating a Word Level Tokenizer and Embedding"""

# Total Words in our Dataset
MAX_TOKENS = 331000 # from the paper

# Creating a TextVectorizer
text_vectorizer = TextVectorization(max_tokens = MAX_TOKENS, output_sequence_length = output_seq_len)

# Since data is large for Colab RAM, we will convert it into batches
train_sentences_dataset = tf.data.Dataset.from_tensor_slices(train_sentences)
train_sentences_dataset = train_sentences_dataset.batch(512).prefetch(tf.data.AUTOTUNE)

text_vectorizer.adapt(train_sentences_dataset)

# How many words in our training vocabulary?
rct_text_vocab = text_vectorizer.get_vocabulary()
print(f"No. of words in vocab: {len(rct_text_vocab)}")
print(f"Most common words in vocab: {rct_text_vocab[:5]}")
print(f"Least common words in data: {rct_text_vocab[-5:]}")

# Creating a custom token embedding
token_embed = Embedding(input_dim = len(rct_text_vocab),
                        output_dim = 512,
                        mask_zero = True,
                        name = "token_embeddings")

"""### Creating a character level tokenizer and embedding"""

# Split sequence-level data splits into character-level data splits
train_chars = [split_chars(sentence) for sentence in train_sentences]
val_chars = [split_chars(sentence) for sentence in val_sentences]
test_chars = [split_chars(sentence) for sentence in test_sentences]

# Average character length in a sentence
char_lens = [len(sentence) for sentence in train_sentences]
output_seq_char_len = int(np.percentile(char_lens, 95))

# Getting all the possible chars in the sentences
import string
alphabet = string.ascii_lowercase + string.digits + string.punctuation

NUM_CHAR_TOKENS = len(alphabet) + 2
char_vectorizer = TextVectorization(max_tokens= NUM_CHAR_TOKENS,
                                 output_sequence_length = output_seq_char_len,
                                 standardize = None,
                                 name = "char_vectorizer")

# Create a char dataset using data api 
train_char_dataset = tf.data.Dataset.from_tensor_slices(train_chars)
train_char_dataset = train_char_dataset.batch(512).prefetch(tf.data.AUTOTUNE)

char_vectorizer.adapt(train_char_dataset)
char_vocab = char_vectorizer.get_vocabulary()
print(f"No of different characters in character voacb: {len(char_vocab)}")
print(f"5 most common characters: {char_vocab[:5]}")
print(f"5 most common characters: {char_vocab[-5:]}")

# Creating an embedding layer
char_embed = Embedding(input_dim = len(char_vocab),
                              output_dim = 25, # this is the size of char_embedding in paper
                              mask_zero = True,
                              name = "char_embed")

"""### One hot Encoding the Line Number and Total Lines"""

# Use tensorflow to create one hot encoded tensors of line number and total lines
train_line_numbers_one_hot = tf.one_hot(train_df["line_number"].to_numpy(), depth=15)
val_line_numbers_one_hot = tf.one_hot(val_df["line_number"].to_numpy(), depth=15)
test_line_numbers_one_hot = tf.one_hot(test_df["line_number"].to_numpy(), depth=15)

train_total_lines_one_hot = tf.one_hot(train_df["total_lines"].to_numpy(), depth=20)
val_total_lines_one_hot = tf.one_hot(val_df["total_lines"].to_numpy(), depth=20)
test_total_lines_one_hot = tf.one_hot(test_df["total_lines"].to_numpy(), depth=20)



# Creating the fast loading prefetch datasets
train_char_token_pos_dataset = create_prefetch_dataset(train_line_numbers_one_hot,
                                                       train_total_lines_one_hot,
                                                       train_sentences,
                                                       train_chars,
                                                       train_labels_one_hot,
                                                       32)

test_char_token_pos_dataset = create_prefetch_dataset(test_line_numbers_one_hot,
                                                       test_total_lines_one_hot,
                                                       test_sentences,
                                                       test_chars,
                                                       test_labels_one_hot,
                                                       32)

val_char_token_pos_dataset = create_prefetch_dataset(val_line_numbers_one_hot,
                                                       val_total_lines_one_hot,
                                                       val_sentences,
                                                       val_chars,
                                                       val_labels_one_hot,
                                                       32)

"""## Creating a Tribrid Model with Custom Token and Word Embeddings"""
# 1. Token Inputs
token_inputs = layers.Input(shape=[], dtype="string", name="token_inputs")
token_vectors = text_vectorizer(token_inputs)
token_embeddings = token_embed(token_vectors)
tokens_bi_lstm = Bidirectional(LSTM(32))(token_embeddings)
token_model = tf.keras.Model(inputs = token_inputs,
                             outputs = tokens_bi_lstm,
                             name = "token_model")

# 2. Character Inputs
char_input = Input(shape=(1,), dtype="string", name="char_inputs")
char_vectors = char_vectorizer(char_input)
char_embeddings = char_embed(char_vectors)
char_bi_lstm = Bidirectional(LSTM(32))(char_embeddings)
char_model = Model(inputs = char_input, outputs = char_bi_lstm, name="char_model")

# 3. Line Numbers
line_numbers_input = Input(shape=(15,), dtype=tf.float32, name="line_number_input")
x = Dense(256, activation="relu")(line_numbers_input)
line_numbers_model = Model(inputs = line_numbers_input, outputs = x, name="line_numbers_model")

# 4. Total Lines
total_lines_input = Input(shape=(20,), dtype=tf.float32, name="total_lines_input")
y = Dense(256, activation="relu")(total_lines_input)
total_lines_model = Model(inputs = total_lines_input, outputs = y, name="total_lines_model")

# 5. Concatenate the token and char into hybrid embeddings
hybrid = Concatenate(name="hybrid_token_char_embeddings")([token_model.output, char_model.output])
z = Dense(256, activation="relu")(hybrid)
hybrid_embeddings = Dropout(0.5)(z)

# 6. Combine positional and hybrid embeddings
tribrid_embeddings = Concatenate(name="tribrid_embeddings")([line_numbers_model.output,
                                                             total_lines_model.output,
                                                             hybrid_embeddings])

# 7. Final Output layer
output = Dense(num_classes, activation="softmax", name="output_layer")(tribrid_embeddings)

# 8. Putting togathere everything
tribrid_model = Model(inputs = [line_numbers_model.input,
                                total_lines_model.input,
                                token_model.input,
                                char_model.input],
                      outputs = output,
                      name = "tribrid_model")

plot_model(tribrid_model, show_shapes=True)

"""### Setting Up Callbacks"""
# Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 3 epochs
early_stopping = EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=3) # if val loss decreases for 3 epochs in a row, stop training

# Create ModelCheckpoint callback to save best model during fine-tuning
checkpoint_path = "/content/gdrive/MyDrive/PUBMED/tribrid_checkpoints/"
model_checkpoint = ModelCheckpoint(checkpoint_path,
                                                      save_best_only=True,
                                                      monitor="val_loss")

# Creating learning rate reduction callback
reduce_lr = ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=2,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)

tribrid_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      metrics = ["accuracy"])

tribrid_model_history = tribrid_model.fit(train_char_token_pos_dataset,
                                          steps_per_epoch = int(0.3 * len(train_char_token_pos_dataset)),
                                          epochs=20,
                                          validation_data = val_char_token_pos_dataset,
                                          validation_steps = int(0.3 * len(val_char_token_pos_dataset)),
                                          callbacks=[early_stopping, model_checkpoint, reduce_lr])

tribrid_model.save("tribrid.h5")
