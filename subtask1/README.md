## Response Selection for Conversation Systems in Tensorflow (DSTC 8)

#### Overview

This code implements a simple dual encoder baseline for the subtask 1 of DSTC-8 [Sentence Selection track](https://github.com/dstc8-track2/NOESIS-II).

This code uses parts of the [work](https://github.com/jdongca2003/next_utterance_selection) from Jianxiong Dong

#### Setup

This code uses Python 3.6 and Tensorflow-GPU 1.6. Clone the repository and install all required packages. It is recommended to use the [Anaconda package manager](https://www.anaconda.com/download/#macos). After installing Anaconda -

```
conda create --name dstc8 python=3.6
source activate dstc7
pip install -r requirements.txt
```

#### Get the data

Make sure you register for the track 2 of DSTC8 to download the data and copy it inside the `data` directory.

#### Prepare the data

Before training, the data needs to converted into a suitable format. The script `convert_dstc8_data.py` can be used to convert data for both advising and ubuntu datasets.

```
python scripts/convert_dstc8_data.py --train_in data/Task_1/ubuntu/task-1.ubuntu.train.json
--train_out data/Task_1/ubuntu/task-1.ubuntu.train.txt
--dev_in data/Task_1/ubuntu/task-1.ubuntu.dev.json
--dev_out data/Task_1/ubuntu/task-1.ubuntu.dev.txt
--answers_file data/Task_1/ubuntu/ubuntu_task_1_candidates.txt
--save_vocab_path data/Task_1/ubuntu/ubuntu_task_1_vocab.txt
```

#### Training

And then train the model

```
python answer_selection/train.py --answer_file data/Task_1/ubuntu/ubuntu_task_1_candidates.txt
--train_file data/Task_1/ubuntu/task-1.ubuntu.train.txt
--embedded_vector_file data/embeddings/glove.6B.100d.txt
--vocab_file data/Task_1/ubuntu/ubuntu_task_1_vocab.txt
--valid_file data/Task_1/ubuntu/task-1.ubuntu.dev.txt
--max_sequence_length 180
--embedding_dim 100
--l2_reg_lambda 0
--dropout_keep_prob 1.0
--batch_size 64
--rnn_size 200
--evaluate_every 2
--char_vocab_file data/embeddings/char_vocab.txt
--max_word_length 18
```

Similar command works for subtask 1 of Advising data as well.

Glove embeddings can be downloaded from [here](https://nlp.stanford.edu/projects/glove/). Embeddings used for the baseline run can be found [here](https://github.com/jdongca2003/next_utterance_selection#Dataset)

#### Model

This baseline model extends the dual-encoder model used [here](http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow).


#### Dual Encoder Baselines (Recall)

Baselines are reported on validation set.

| Dataset           | 1 in 100 R@1 | 1 in 100 R@2 | 1 in 100 R@5 | 1 in 100 R@10 | MRR |
| :---------------: | :-------------: | :-----------: |:----------: | :---------: | :---------: |
| Ubuntu - Subtask 1 | 21.16% | 29.03% | 42.11% | 56.53% | 0.3249 |
| Advising - Subtask 1 | 22.18% | 33.60% | 49.31% | 62.20% | 0.3551 |
