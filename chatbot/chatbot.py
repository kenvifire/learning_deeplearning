# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import re
import time

lines = open('move_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('move_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')


id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]]  = _line[4]

conversation_ids = []

for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversation_ids.append(_conversation.split(","))

questions = []
answers = []

for conversation in conversation_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i + 1]])


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"[-()\"E/@;:<>{}+-~|.?,]", "", text)
    return text

clean_questions = []

for question in questions:
    clean_questions.append(clean_text(questions))

clean_answers = []
for answers in answers:
    clean_answers.append(clean_text(answers))

word2count = {}

for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

threshold = 20
questionwords2int = {}
word_num = 0
for word, count in word2count.item():
    if count >= 20:
        questionwords2int[word] = word_num
        word_num += 1

answerwords2int = {}
word_num = 0
for word, count in word2count.item():
    if count >= 20:
        answerwords2int[word] = word_num
        word_num += 1

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

for token in tokens:
    questionwords2int[token] = len(questionwords2int) + 1


for token in tokens:
    answerwords2int[token] = len(answerwords2int) + 1

answersints2word = {w_i: w for w, w_i in answerwords2int.items()}
questionsints2word = {}

for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

questions_into_int = []

for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionwords2int:
            ints.append(questionwords2int['<OUT>'])
        else:
            ints.append(questionwords2int[word])
    questions_into_int.append(ints)

answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerwords2int:
            ints.append(answerwords2int['<OUT>'])
        else:
            ints.append(answerwords2int[word])
    answers_into_int.append(ints)

sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])


#############
def model_inputs():
    inputs = ft.placeholder(tf.int32, [None, None], name = 'input')
    targets = ft.placeholder(tf.int32, [None, None], name = 'targets')
    learning_rate = ft.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = ft.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, learning_rate, keep_prob


def preprocess_targets(targets, word2int, batch_size):
    left_size = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_size = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    preprocessed_targets = tf.concat([left_size, right_size], 1)
    return preprocessed_targets


def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.contrib.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                            cell_bw = encoder_cell,
                                                            sequence_length = sequence_length,
                                                            inputs = rnn_inputs,
                                                            dtype = tf.float32)
    return encoder_state


def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                    attention_option = 'bahdanau',
                                                                                                                                    units=)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoer_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                             training_decoder_function,
                                                                                                             decoder_embedded_input,
                                                                                                             sequence_length,
                                                                                                             scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)

    return output_function(decoder_output_dropout)


def decode_test_set(encoder_state, decoder_cell, decoder_embedded_matrix, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                                                                                                                    attention_option = 'bahdanau',
                                                                                                                                    units=)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embedded_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, _, _= tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                   test_decoder_function,
                                                                   scope = decoding_scope)

    return test_predictions

def decoder_rnn(decoder_embeded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output = lambda x: tf.contrib.layers.fully_connected(x,
                                                             num_words,
                                                             None,
                                                             scope = decoding_scope,
                                                             weights_initializers = weights,
                                                             biases = biases)
        training_predictions = decode_test_set(encoder_state,
                                               decoder_cell,
                                               decoder_embeded_input,
                                               sequence_length,
                                               decoding_scope,
                                               output_function,
                                               keep_prob,
                                               batch_size)

        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['SOS'],
                                           word2int['EOS'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions


def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words,question_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input  = tf.contrib.layers.embed_sequence(inputs,
                                                               answers_num_words + 1,
                                                               encoder_embedding_size,
                                                               initializer = tf.random_unifor_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionwords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([question_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embeded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embeded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         question_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionwords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions


### PART 3
epochs = 100
batch_size = 64
num_layers - 3








