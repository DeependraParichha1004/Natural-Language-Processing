import numpy as np
import pandas as pd


def generate_dictinoary_data(text):
    word_to_index = dict()
    index_to_word = dict()
    corpus = []
    count = 0
    vocab_size = 0

    for row in text:
        for word in row.split():
            word = word.lower()
            corpus.append(word)
            if word_to_index.get(word) == None:
                word_to_index.update({word: count})
                index_to_word.update({count: word})
                count += 1
    vocab_size = len(word_to_index)
    length_of_corpus = len(corpus)

    return word_to_index, index_to_word, corpus, vocab_size, length_of_corpus








def get_one_hot_vectors(target_word, context_words, vocab_size, word_to_index):
    # Create an array of size = vocab_size filled with zeros
    trgt_word_vector = np.zeros(vocab_size)

    # Get the index of the target_word according to the dictionary word_to_index.
    # If target_word = best, the index according to the dictionary word_to_index is 0.
    # So the one hot vector will be [1, 0, 0, 0, 0, 0, 0, 0, 0]
    index_of_word_dictionary = word_to_index.get(target_word)

    # Set the index to 1
    trgt_word_vector[index_of_word_dictionary] = 1

    # Repeat same steps for context_words but in a loop
    ctxt_word_vector = np.zeros(vocab_size)

    for word in context_words:
        index_of_word_dictionary = word_to_index.get(word)
        ctxt_word_vector[index_of_word_dictionary] = 1

    return trgt_word_vector, ctxt_word_vector


# Note : Below comments for trgt_word_index, ctxt_word_index are with the above sample text for understanding the code flow

def generate_training_data(corpus, window_size, vocab_size, word_to_index, length_of_corpus, sample=None):
    training_data = []
    training_sample_words = []
    for i, word in enumerate(corpus):

        index_target_word = i
        target_word = word
        context_words = []

        # when target word is the first word
        if i == 0:

            # trgt_word_index:(0), ctxt_word_index:(1,2)
            context_words = [corpus[x] for x in range(i + 1, window_size + 1)]

            # when target word is the last word
        elif i == len(corpus) - 1:

            # trgt_word_index:(9), ctxt_word_index:(8,7), length_of_corpus = 10
            context_words = [corpus[x] for x in range(length_of_corpus - 2, length_of_corpus - 2 - window_size, -1)]

        # When target word is the middle word
        else:

            # Before the middle target word
            before_target_word_index = index_target_word - 1
            for x in range(before_target_word_index, before_target_word_index - window_size, -1):
                if x >= 0:
                    context_words.extend([corpus[x]])

            # After the middle target word
            after_target_word_index = index_target_word + 1
            for x in range(after_target_word_index, after_target_word_index + window_size):
                if x < len(corpus):
                    context_words.extend([corpus[x]])

        trgt_word_vector, ctxt_word_vector = get_one_hot_vectors(target_word, context_words, vocab_size, word_to_index)
        training_data.append([trgt_word_vector, ctxt_word_vector])

        if sample is not None:
            training_sample_words.append([target_word, context_words])

    return training_data, training_sample_words





def forward_prop(weight_inp_hidden, weight_hidden_output, target_word_vector):
    # target_word_vector = x , weight_inp_hidden =  weights for input layer to hidden layer
    hidden_layer = np.dot(weight_inp_hidden.T, target_word_vector)

    # weight_hidden_output = weights for hidden layer to output layer
    u = np.dot(weight_hidden_output.T, hidden_layer)

    y_predicted = softmax(u)

    return y_predicted, hidden_layer, u


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def backward_prop(weight_inp_hidden, weight_hidden_output, total_error, hidden_layer, target_word_vector,
                  learning_rate):
    dl_weight_inp_hidden = np.outer(target_word_vector, np.dot(weight_hidden_output, total_error.T))
    dl_weight_hidden_output = np.outer(hidden_layer, total_error)

    # Update weights
    weight_inp_hidden = weight_inp_hidden - (learning_rate * dl_weight_inp_hidden)
    weight_hidden_output = weight_hidden_output - (learning_rate * dl_weight_hidden_output)

    return weight_inp_hidden, weight_hidden_output

def calculate_error(y_pred, context_words):

    total_error = [None] * len(y_pred)
    index_of_1_in_context_words = {}

    for index in np.where(context_words == 1)[0]:
        index_of_1_in_context_words.update({index: 'yes'})

    number_of_1_in_context_vector = len(index_of_1_in_context_words)

    for i, value in enumerate(y_pred):

        if index_of_1_in_context_words.get(i) != None:
            total_error[i] = (value - 1) + ((number_of_1_in_context_vector - 1) * value)
        else:
            total_error[i] = (number_of_1_in_context_vector * value)

    return np.array(total_error)



def calculate_loss(u, ctx):
    sum_1 = 0
    for index in np.where(ctx == 1)[0]:
        sum_1 = sum_1 + u[index]

    sum_1 = -sum_1
    sum_2 = len(np.where(ctx == 1)[0]) * np.log(np.sum(np.exp(u)))

    total_loss = sum_1 + sum_2
    return total_loss





def train(word_embedding_dimension, window_size, epochs, training_data, learning_rate):
    weights_input_hidden = np.random.uniform(-1, 1, (vocab_size, word_embedding_dimension))
    weights_hidden_output = np.random.uniform(-1, 1, (word_embedding_dimension, vocab_size))

    # For analysis purposes
    epoch_loss = []
    weights_1 = []
    weights_2 = []

    for epoch in range(epochs):
        loss = 0

        for target, context in training_data:
            y_pred, hidden_layer, u = forward_prop(weights_input_hidden, weights_hidden_output, target)

            total_error = calculate_error(y_pred, context)

            weights_input_hidden, weights_hidden_output = backward_prop(
                weights_input_hidden, weights_hidden_output, total_error, hidden_layer, target, learning_rate
            )

            loss_temp = calculate_loss(u, context)
            loss += loss_temp

        epoch_loss.append(loss)
        weights_1.append(weights_input_hidden)
        weights_2.append(weights_hidden_output)


    return epoch_loss, np.array(weights_1), np.array(weights_2)



epochs = 100
text = ["machine learning engineers can build great data models","the more data you have the better your model","these predictions sound right, but it is all about your data","your data can provide great value"]
word_to_index,index_to_word,corpus,vocab_size,length_of_corpus = generate_dictinoary_data(text)
training_data,training_sample_words = generate_training_data(corpus,5,vocab_size,word_to_index,length_of_corpus)
loss,w_1,w_2=train(5,5,epochs,training_data,0.01)

b=w_1[word_to_index['machine']]
print(b)

c=w_1[word_to_index['learning']]
print(c)
