# Copyright (c) 2019 IBM Corp. Intellectual Property. All rights reserved.
# Copyright (c) 2017 AT&T Intellectual Property. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by IBM Corp. to add support for DSTC8 Track 2
#
# ==============================================================================

import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS


def get_embeddings(vocab):
    initializer = load_word_embeddings(vocab, FLAGS.embedding_dim)
    return tf.constant(initializer, name="word_embedding")


def get_char_embedding(charVocab):
    char_size = len(charVocab)
    embeddings = np.zeros((char_size, char_size), dtype='float32')
    for i in range(1, char_size):
        embeddings[i, i] = 1.0

    return tf.constant(embeddings, name="word_char_embedding")


def load_embed_vectors(fname, dim):
    vectors = {}
    for line in open(fname, 'rt', encoding='utf-8'):
        items = line.strip().split(' ')
        if len(items[0]) <= 0:
            continue
        vec = [float(items[i]) for i in range(1, dim + 1)]
        vectors[items[0]] = vec

    return vectors


def load_word_embeddings(vocab, dim):
    vectors = load_embed_vectors(FLAGS.embedded_vector_file, dim)
    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size + 1, dim), dtype='float32')
    for word, code in vocab.items():
        if word in vectors:
            embeddings[code] = vectors[word]
        else:
            embeddings[code] = np.random.uniform(-0.25, 0.25, dim)

    return embeddings


def lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, scope, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse) as vs:
        fw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_prob)
        bw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_prob)
        rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                  inputs=inputs,
                                                                  sequence_length=input_seq_len,
                                                                  dtype=tf.float32)
        return rnn_outputs, rnn_states


class DualEncoder(object):
    def __init__(
            self, sequence_length, vocab_size, embedding_size, vocab, rnn_size, maxWordLength, charVocab,
            l2_reg_lambda=0.0):
        # question
        self.question = tf.placeholder(tf.int32, [None, sequence_length], name="question")
        # answer
        self.answer = tf.placeholder(tf.int32, [None, sequence_length], name="answer")

        self.target = tf.placeholder(tf.float32, [None], name="target")

        self.target_loss_weight = tf.placeholder(tf.float32, [None], name="target_weight")

        self.question_len = tf.placeholder(tf.int32, [None], name="question_len")
        self.answer_len = tf.placeholder(tf.int32, [None], name="answer_len")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.q_charVec = tf.placeholder(tf.int32, [None, sequence_length, maxWordLength], name="question_char")
        self.q_charLen = tf.placeholder(tf.int32, [None, sequence_length], name="question_char_len")

        self.a_charVec = tf.placeholder(tf.int32, [None, sequence_length, maxWordLength], name="answer_char")
        self.a_charLen = tf.placeholder(tf.int32, [None, sequence_length], name="answer_char_len")

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = get_embeddings(vocab)
            question_embedded = tf.nn.embedding_lookup(W, self.question)
            answer_embedded = tf.nn.embedding_lookup(W, self.answer)

        with tf.device('/cpu:0'), tf.name_scope('char_embedding'):
            char_W = get_char_embedding(charVocab)
            # [batch_size, q_len, maxWordLength, char_dim]
            question_char_embedded = tf.nn.embedding_lookup(char_W, self.q_charVec)

            # [batch_size, a_len, maxWordLength, char_dim]
            answer_char_embedded = tf.nn.embedding_lookup(char_W, self.a_charVec)

        charRNN_size = 40
        charRNN_name = "char_RNN"
        char_dim = question_char_embedded.get_shape()[3].value
        question_char_embedded = tf.reshape(question_char_embedded, [-1, maxWordLength, char_dim])
        question_char_len = tf.reshape(self.q_charLen, [-1])
        answer_char_embedded = tf.reshape(answer_char_embedded, [-1, maxWordLength, char_dim])
        answer_char_len = tf.reshape(self.a_charLen, [-1])

        char_rnn_output1, char_rnn_states1 = lstm_layer(question_char_embedded, question_char_len, charRNN_size,
                                                        self.dropout_keep_prob, charRNN_name, scope_reuse=False)
        char_rnn_output2, char_rnn_states2 = lstm_layer(answer_char_embedded, answer_char_len, charRNN_size,
                                                        self.dropout_keep_prob, charRNN_name, scope_reuse=True)

        question_char_state = tf.concat(axis=1, values=[char_rnn_states1[0].h, char_rnn_states1[1].h])
        char_embed_dim = 2 * charRNN_size
        question_char_state = tf.reshape(question_char_state, [-1, sequence_length, char_embed_dim])

        answer_char_state = tf.concat(axis=1, values=[char_rnn_states2[0].h, char_rnn_states2[1].h])
        answer_char_state = tf.reshape(answer_char_state, [-1, sequence_length, char_embed_dim])

        question_embedded = tf.concat(axis=2, values=[question_embedded, question_char_state])
        answer_embedded = tf.concat(axis=2, values=[answer_embedded, answer_char_state])

        # Build the Context Encoder RNN
        with tf.variable_scope("context-rnn") as vs:
            # We use an LSTM Cell
            cell_context = tf.nn.rnn_cell.LSTMCell(
                rnn_size,
                forget_bias=2.0,
                use_peepholes=True,
                state_is_tuple=True)

            # Run context through the RNN
            context_encoded_outputs, context_encoded_states = tf.nn.dynamic_rnn(cell_context, question_embedded,
                                                                                self.question_len, dtype=tf.float32)

        # Build the Utterance Encoder RNN
        with tf.variable_scope("utterance-rnn") as vs:
            # We use an LSTM Cell
            cell_utterance = tf.nn.rnn_cell.LSTMCell(
                rnn_size,
                forget_bias=2.0,
                use_peepholes=True,
                state_is_tuple=True)

            # Run the utterance through the RNN
            utterance_encoded_outputs, utterance_encoded_states = tf.nn.dynamic_rnn(cell_utterance, answer_embedded,
                                                                                    self.answer_len, dtype=tf.float32)

        with tf.variable_scope("prediction") as vs:
            M = tf.get_variable("M",
                                shape=[rnn_size, rnn_size],
                                initializer=tf.truncated_normal_initializer())

            # "Predict" a  response: c * M
            generated_response = tf.matmul(context_encoded_states.h, M)
            generated_response = tf.expand_dims(generated_response, 2)
            encoding_utterance = tf.expand_dims(utterance_encoded_states.h, 2)

            # Dot product between generated response and actual response
            # (c * M) * r
            logits = tf.matmul(generated_response, encoding_utterance, True)
            logits = tf.squeeze(logits, [2])
            logits = tf.squeeze(logits, [1])

            # Apply sigmoid to convert logits to probabilities
            self.probs = tf.sigmoid(logits, name="prob")

            # Calculate the binary cross-entropy loss
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(self.target))

        # Mean loss across the batch of examples
        self.mean_loss = tf.reduce_mean(losses, name="mean_loss")

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.sign(self.probs - 0.5), tf.sign(self.target - 0.5))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
