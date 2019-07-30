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

import sys, os
import time
import datetime
from answer_selection import data_helpers
from answer_selection.model import DualEncoder

from collections import defaultdict
from answer_selection import metrics

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.000005, "L2 regularizaion lambda (default: 0.0)")

tf.flags.DEFINE_string("answers_file", "", "tokenized answers")
tf.flags.DEFINE_string("train_file", "", "train file containing (question, positive and negative answer ids)")
tf.flags.DEFINE_string("embedded_vector_file", "", "pre-trained embedded word vector")
tf.flags.DEFINE_string("vocab_file", "", "vocabulary file (map word to integer)")
tf.flags.DEFINE_string("valid_file", "", "validation file containg (question, positive and negative response ids")
tf.flags.DEFINE_integer("max_sequence_length", 200, "max sequence length")
tf.flags.DEFINE_integer("rnn_size", 200, "number of RNN units")
tf.flags.DEFINE_string("char_vocab_file", "", "vocabulary file (map char to integer)")
tf.flags.DEFINE_integer("max_word_length", 18, "max word length")

# Training parameters

tf.flags.DEFINE_integer("batch_size", 1024, "Batch Size (default: 64)")

tf.flags.DEFINE_integer("num_epochs", 5000000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("load_checkpoint", False, "Load from checkpoint")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr.upper(), value))
print("")


def train():
    graph = tf.Graph()
    with graph.as_default():
      with tf.device("/gpu:0"):
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = DualEncoder(
                sequence_length=FLAGS.max_sequence_length,
                vocab_size=len(vocab),
                embedding_size=FLAGS.embedding_dim,
                vocab=vocab,
                rnn_size=FLAGS.rnn_size,
                maxWordLength=FLAGS.max_word_length,
                charVocab=charVocab,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            starter_learning_rate = 0.001
            target_loss_weight = [1.0, 1.0]
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                                       5000, 0.96, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(model.mean_loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            best_mrr = 0.0
            batches = data_helpers.batch_iter(train_dataset, FLAGS.batch_size, FLAGS.num_epochs, target_loss_weight,
                                              FLAGS.max_sequence_length, charVocab, FLAGS.max_word_length, shuffle=True)
            for batch in batches:
                x_question, x_answer, x_question_len, x_answer_len, x_target, x_target_weight, id_pairs, x_q_char, x_q_len, x_a_char, x_a_len = batch
                feed_dict = {
                    model.question: x_question,
                    model.answer: x_answer,
                    model.question_len: x_question_len,
                    model.answer_len: x_answer_len,
                    model.target: x_target,
                    model.target_loss_weight: x_target_weight,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    model.q_charVec: x_q_char,
                    model.q_charLen: x_q_len,
                    model.a_charVec: x_a_char,
                    model.a_charLen: x_a_len
                }
                train_step(sess, train_op, global_step, model, feed_dict)
                current_step = tf.train.global_step(sess, global_step)
                if current_step > 10 and current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    valid_mrr = dev_step(sess, model, target_loss_weight)
                    if valid_mrr > best_mrr:
                        best_mrr = valid_mrr
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))


def train_step(sess, train_op, global_step, model, feed_dict):
    """
    A single training step
    """
    _, step, loss, accuracy, predicted_prob = sess.run(
        [train_op, global_step, model.mean_loss, model.accuracy, model.probs],
        feed_dict)

    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))


def dev_step(sess, model, target_loss_weight):
    results = defaultdict(list)
    num_test = 0
    num_correct = 0.0
    test_batches = data_helpers.batch_iter(valid_dataset, FLAGS.batch_size, 1, target_loss_weight, FLAGS.max_sequence_length, charVocab, FLAGS.max_word_length, shuffle=True)
    for test_batch in test_batches:
        x_question, x_answer, x_question_len, x_answer_len, x_target, x_target_weight, id_pairs, x_q_char, x_q_len, x_a_char, x_a_len = test_batch
        feed_dict = {
          model.question: x_question,
          model.answer: x_answer,
          model.question_len: x_question_len,
          model.answer_len: x_answer_len,
          model.target: x_target,
          model.target_loss_weight: x_target_weight,
          model.dropout_keep_prob: 1.0,
          model.q_charVec: x_q_char,
          model.q_charLen: x_q_len,
          model.a_charVec: x_a_char,
          model.a_charLen: x_a_len
        }
        batch_accuracy, predicted_prob = sess.run([model.accuracy, model.probs], feed_dict)
        num_test += len(predicted_prob)
        if num_test % 1000 == 0:
            print(num_test)

        num_correct += len(predicted_prob) * batch_accuracy
        for i, prob_score in enumerate(predicted_prob):
            question_id, answer_id, label = id_pairs[i]
            results[question_id].append((answer_id, label, prob_score))

    #calculate top-1 precision
    print('num_test_samples: {}  test_accuracy: {}'.format(num_test, num_correct/num_test))
    accu, precision, recall, f1, loss = metrics.classification_metrics(results)
    print('Accuracy: {}, Precision: {}  Recall: {}  F1: {} Loss: {}'.format(accu, precision, recall, f1, loss))

    mrr = metrics.mean_reciprocal_rank(results)
    top_1_precision = metrics.top_k_precision(results, k=1)
    top_2_precision = metrics.top_k_precision(results, k=2)
    top_5_precision = metrics.top_k_precision(results, k=5)
    top_10_precision = metrics.top_k_precision(results, k=10)
    total_valid_query = metrics.get_num_valid_query(results)

    print('MRR (mean reciprocal rank): {}\tTop-1 precision: {}\tNum_query: {}\n'.format(mrr, top_1_precision, total_valid_query))
    print('Top-2 precision: {}'.format(top_2_precision))
    print('Top-5 precision: {}'.format(top_5_precision))
    print('Top-10 precision: {}'.format(top_10_precision))

    return mrr


if __name__=="__main__":
    # Load fixtures
    print("Loading data...")

    vocab = data_helpers.loadVocab(FLAGS.vocab_file)
    charVocab = data_helpers.loadCharVocab(FLAGS.char_vocab_file)

    answer_data = data_helpers.loadAnswers(FLAGS.answers_file, vocab, FLAGS.max_sequence_length)
    train_dataset = data_helpers.loadDataset(FLAGS.train_file, vocab, FLAGS.max_sequence_length, answer_data,
                                             do_label_smoothing=True)
    print('train_pairs: {}'.format(len(train_dataset)))
    valid_dataset = data_helpers.loadDataset(FLAGS.valid_file, vocab, FLAGS.max_sequence_length, answer_data,
                                            do_label_smoothing=False)
    print('dev_pairs: {}'.format(len(valid_dataset)))

    train()