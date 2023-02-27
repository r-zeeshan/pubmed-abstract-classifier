from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow as tf

# Creating the function to read lines from the document
def get_lines(filename):
    """
    Reads filename (a text file) and returns the lines of text as a list.

    Args:
        filename: a string contaning the target filepath.

    Returns:
        A list of strings with one string per line from the target filename.
    """

    with open(filename) as f:
        return f.readlines()



def split_chars(text):
    return " ".join(list(text))


# Creating a function to preprocess the data
def preprocess_text(filename):
    """
    Args: 
        filename : filename: a string contaning the target filepath.
    Returns:
        A list of dictionaries, of abstract line data.
    """

    input_lines = get_lines(filename)
    abstract_lines = ""
    abstract_samples = []

    # Loop through each line in the target file
    for line in input_lines:
        if line.startswith("###"):
            abstract_id = line
            abstract_lines = ""
        elif line.isspace():
            abstract_line_split = abstract_lines.splitlines()

            # Iterate through each line in a single abstract and count them at the same time
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}
                target_text_split = abstract_line.split("\t") # split the label from text
                line_data["target"] = target_text_split[0]
                line_data["text"] = target_text_split[1].lower()
                line_data["line_number"] = abstract_line_number
                line_data["total_lines"] = len(abstract_line_split) - 1
                abstract_samples.append(line_data)
        else:
            abstract_lines += line

    return abstract_samples 


def visualize_pred_sequence_labels(abstract, model, label_encoder):
    '''
    Takes in a string abstract and makes predictions on the sequence labels for each line of the abstract

    Arguments: 
    ----------
      - abstract : string of abstract text
      - model : the trained model on the same data format (line_numbers, total_lines, sentences, characters)
      - label_encoder : the label encoder used to encode the classes 

    Returns:
    --------
      Prints out the predicted label and the corresponding sequence/text 
    '''

    # Create list of lines from abstract string
    abstract_lines = abstract.split('. ')

    # Get total number of lines 
    total_lines_in_sample = len(abstract_lines)

    # Loop through each line in the abstract and create a list of dictionaries containing features 
    sample_lines = []
    for i, line in enumerate(abstract_lines):
        sample_dict = {}
        sample_dict['text'] = str(line)
        sample_dict['line_number'] = i 
        sample_dict['total_lines'] = total_lines_in_sample - 1 
        sample_lines.append(sample_dict)

    # Get all line number and total lines numbers then one hot encode them 
    abstract_line_numbers = [line['line_number'] for line in sample_lines]
    abstract_total_lines = [line['total_lines'] for line in sample_lines]

    abstract_line_numbers_one_hot = tf.one_hot(abstract_line_numbers , depth=15)
    abstract_total_lines_one_hot = tf.one_hot(abstract_total_lines , depth=20)

    # Split the lines into characters 
    abstract_chars = [split_chars(sentence) for sentence in abstract_lines]

    # Making prediction on sample features
    abstract_pred_probs = model.predict(x=(abstract_line_numbers_one_hot, 
                                           abstract_total_lines_one_hot, 
                                           tf.constant(abstract_lines), 
                                           tf.constant(abstract_chars)))
  
    # Turn prediction probs to pred class 
    abstract_preds = tf.argmax(abstract_pred_probs , axis=1)
  
    # Prediction class integers into string class name 
    abstract_pred_classes = [label_encoder.classes_[i] for i in abstract_preds]

    # Prints out the abstract lines and the predicted sequence labels 
    for i, line in enumerate(abstract_lines):
        print(f'{abstract_pred_classes[i]}:  {line}\n')


def create_prefetch_dataset(line_numbers, total_lines, sentences, chars, labels, batch_size):
    """
    Args:
        line_numbers: line numbers ONE HOT ENCODED
        total_lines: total_lines ONE HOT ENCODED
        sentences: list of sentences
        chars: list of characters
        batch_size: Number of batches
        labels: Labels ONE HOT ENCODED

    Returns: 
        Prefetch tensorflow dataset for efficient performance
    """

    char_token_pos_data = tf.data.Dataset.from_tensor_slices((line_numbers,
                                                              total_lines,
                                                              sentences,
                                                              chars))
    
    char_token_pos_labels = tf.data.Dataset.from_tensor_slices(labels)

    char_token_pos_dataset = tf.data.Dataset.zip((char_token_pos_data, char_token_pos_labels))

    char_token_pos_dataset = char_token_pos_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return char_token_pos_dataset


def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.
  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array
  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results
