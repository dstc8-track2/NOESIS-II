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
import os, sys

from answer_selection import data_helpers
from collections import defaultdict
import operator
from answer_selection import metrics

# Data Parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_integer("max_sequence_length", 200, "max sequence length")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_string("answer_file", "", "tokenized answers")
tf.flags.DEFINE_string("test_file", "", "test file containing (question, positive and negative answer ids)")
tf.flags.DEFINE_string("vocab_file", "", "vocabulary file (map word to integer)")
tf.flags.DEFINE_string("output_file", "", "prediction output file")

tf.flags.DEFINE_string("char_vocab_file", "", "vocabulary file (map char to integer)")
tf.flags.DEFINE_integer("max_word_length", 18, "max word length")


FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr.upper(), value))
print("")

vocab = data_helpers.loadVocab(FLAGS.vocab_file)
print(len(vocab))

charVocab = data_helpers.loadCharVocab(FLAGS.char_vocab_file)


SEQ_LEN = FLAGS.max_sequence_length
answer_data = data_helpers.loadAnswers(FLAGS.answer_file, vocab, SEQ_LEN)
test_dataset = data_helpers.loadDataset(FLAGS.test_file, vocab, SEQ_LEN, answer_data)


target_loss_weight=[1.0,1.0]

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(os.path.normpath(FLAGS.checkpoint_dir))
print(FLAGS.checkpoint_dir)
print(checkpoint_file)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        question_x = graph.get_operation_by_name("question").outputs[0]
        answer_x   = graph.get_operation_by_name("answer").outputs[0]

        question_len = graph.get_operation_by_name("question_len").outputs[0]
        answer_len = graph.get_operation_by_name("answer_len").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        q_char_feature = graph.get_operation_by_name("question_char").outputs[0]
        q_char_len     = graph.get_operation_by_name("question_char_len").outputs[0]

        a_char_feature =  graph.get_operation_by_name("answer_char").outputs[0]
        a_char_len     = graph.get_operation_by_name("answer_char_len").outputs[0]

        # Tensors we want to evaluate
        prob = graph.get_operation_by_name("prediction/prob").outputs[0]

        results = defaultdict(list)
        num_test = 0
        test_batches = data_helpers.batch_iter(test_dataset, FLAGS.batch_size, 1, target_loss_weight, SEQ_LEN, charVocab, FLAGS.max_word_length, shuffle=False)
        for test_batch in test_batches:
            batch_question, batch_answer, batch_question_len, batch_answer_len, batch_target, batch_target_weight, batch_id_pairs, x_q_char, x_q_len, x_a_char, x_a_len = test_batch
            feed_dict = {
                question_x: batch_question,
                answer_x: batch_answer,
                question_len: batch_question_len,
                answer_len: batch_answer_len,
                dropout_keep_prob: 1.0,
                q_char_feature: x_q_char,
                q_char_len: x_q_len,
                a_char_feature: x_a_char,
                a_char_len: x_a_len
            }
            predicted_prob = sess.run(prob, feed_dict)
            num_test += len(predicted_prob)
            print('num_test_sample={}'.format(num_test))
            for i, prob_score in enumerate(predicted_prob):
                qid, aid, label = batch_id_pairs[i]
                results[qid].append((aid, label, prob_score))

accu, precision, recall, f1, loss = metrics.classification_metrics(results)
print('Accuracy: {}, Precision: {}  Recall: {}  F1: {} Loss: {}'.format(accu, precision, recall, f1, loss))

mrr = metrics.mean_reciprocal_rank(results)
top_1_precision = metrics.top_k_precision(results, k=1)
top_2_precision = metrics.top_k_precision(results, k=2)
top_5_precision = metrics.top_k_precision(results, k=5)
top_10_precision = metrics.top_k_precision(results, k=10)
total_valid_query = metrics.get_num_valid_query(results)
print('MRR (mean reciprocal rank): {}\tTop-1 recall: {}\tNum_query: {}'.format(mrr, top_1_precision, total_valid_query))
print('Top-2 recall: {}'.format(top_2_precision))
print('Top-5 recall: {}'.format(top_5_precision))
print('Top-10 recall: {}'.format(top_10_precision))


out_path = FLAGS.output_file
print("Saving evaluation to {}".format(out_path))
with open(out_path, 'w') as f:
    f.write("query_id\tdocument_id\tscore\trank\trelevance\n")
    for qid, v in results.items():
        v.sort(key=operator.itemgetter(2), reverse=True)
        for i, rec in enumerate(v):
            aid, label, prob_score = rec
            rank = i+1
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(qid, aid, prob_score, rank, label))
