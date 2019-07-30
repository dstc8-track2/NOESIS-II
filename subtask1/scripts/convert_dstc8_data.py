import os
import ijson
import random
import tensorflow as tf

tf.flags.DEFINE_integer("min_word_frequency", 1, "Minimum frequency of words in the vocabulary")
tf.flags.DEFINE_integer("max_sentence_len", 180, "Maximum Sentence Length")
tf.flags.DEFINE_integer("random_seed", 42, "Seed for sampling negative training examples")

tf.flags.DEFINE_string("train_in", "data/Task_1/ubuntu/task-1.ubuntu.train.json", "Path to input data file")
tf.flags.DEFINE_string("train_out", "data/Task_1/ubuntu/task-1.ubuntu.train.txt", "Path to output train file")

tf.flags.DEFINE_string("dev_in", "data/Task_1/ubuntu/task-1.ubuntu.dev.json", "Path to dev data file")
tf.flags.DEFINE_string("dev_out", "data/Task_1/ubuntu/task-1.ubuntu.dev.txt", "Path to output dev file")

tf.flags.DEFINE_string("answers_file", "data/Task_1/ubuntu/ubuntu_task_1_candidates.txt", "Path to write answers file")
tf.flags.DEFINE_string("save_vocab_path", 'data/Task_1/ubuntu/ubuntu_task_1_vocab.txt', "Path to save vocabulary txt file")

## Once we have the test_set, we need to add test_file here too!
# tf.flags.DEFINE_string("test_in", "data/Task_1/ubuntu/..", "Path to test data file")
# tf.flags.DEFINE_string("test_out", "data/Task_1/ubuntu/..", "Path to output test file")
# tf.flags.DEFINE_string("test_answers_file", "data/Task_1/ubuntu/..", "Path to write test answers file")

FLAGS = tf.flags.FLAGS

def tokenizer_fn(iterator):
    return (x.replace('?', ' ').replace('.', ' ').replace(',', ' ').replace('*', ' ').replace('=', ' ').split(" ") for x in iterator)

def process_dialog(dialog, train=False, positive=True, all_negative=False, seed=42):
    row = []

    context = get_context(dialog)
    row.append(context)

    # Get correct answer and target id
    if len(dialog['options-for-correct-answers']) == 0:
        correct_answer = {}
        correct_answer['utterance'] = "None"
        target_id = "NONE"
    else:
        correct_answer = dialog['options-for-correct-answers'][0]
        target_id = correct_answer['candidate-id']

    # Create a list of all negative answers
    negative_answers = []
    for i, utterance in enumerate(dialog['options-for-next']):
        if utterance['candidate-id'] != target_id:
            negative_answers.append(utterance)

    if len(negative_answers) < 100:
        none = {'utterance': "None",
                'candidate-id': "NONE"}
        negative_answers.append(none)

    # If this is a training sample (train=True) and positive=True return the correct response
    if train and positive:
        row.append(correct_answer['utterance'] + " __eou__ ")
        return row
    # If this is a training sample and positive=False return a randomly sampled from list of negative answers
    elif train and not positive and not all_negative:
        random.seed(seed)
        negative_answer = random.choice(negative_answers)
        row.append(negative_answer['utterance'] + " __eou__ ")
        return row
    # If all_negative = True, return all negative options
    elif train and all_negative:
        rows = []
        for option in negative_answers:
            row = []
            row.append(context)
            row.append(option['utterance'] + " __eou__")
            rows.append(row)
        return rows

    return row

def get_dialogs(filename):
    rows = []
    with open(filename, 'rb') as f:
        json_data = ijson.items(f, 'item')
        for entry in json_data:
            rows.append(process_dialog(entry, train=True, positive=True))
            rows.extend(process_dialog(entry, train=True, positive=False, all_negative=True))
    return rows

def create_utterance_iter(input_iter):
    for row in input_iter:
        all_utterances = []
        context = row[0]
        next_utterances = row[1:]
        all_utterances.append(context)
        all_utterances.extend(next_utterances)
        for utterance in all_utterances:
            yield utterance

def create_vocab(input_iter, min_frequency):
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        FLAGS.max_sentence_len,
        min_frequency=min_frequency,
        tokenizer_fn=tokenizer_fn)
    vocab_processor.fit(input_iter)
    return vocab_processor

def write_vocabulary(vocab_processor, outfile):
    vocab_size = len(vocab_processor.vocabulary_)
    count = 0
    with open(outfile, "w") as vocabfile:
        for id in range(vocab_size):
            word = vocab_processor.vocabulary_._reverse_mapping[id]
            if word == '':
                continue
            vocabfile.write(str(count) + "\t" + word.replace("\n", "") + "\n")
            count += 1
        vocabfile.write(str(count) + "\t" + "UNKNOWN" + "\n")
    print("Saved vocabulary to {}".format(outfile))

# TODO - Once we have the test_set, we need to add test_file here too!
def create_answers_file(train_file, dev_file, answers_file):
    answers = set()
    with open(train_file, 'rb') as f:
        json_data = ijson.items(f, 'item')
        for entry in json_data:
            if len(entry['options-for-correct-answers']) == 0:
                correct_answer = {}
                correct_answer['utterance'] = "None"
            else:
                correct_answer = entry['options-for-correct-answers'][0]
            answer = correct_answer['utterance'] + " __eou__ "
            answers.add(answer.strip())

            for i, utterance in enumerate(entry['options-for-next']):
                answer = utterance['utterance'] + " __eou__ "
                answers.add(answer.strip())

    with open(dev_file, 'rb') as f:
        json_data = ijson.items(f, 'item')
        for entry in json_data:
            if len(entry['options-for-correct-answers']) == 0:
                correct_answer = {}
                correct_answer['utterance'] = "None"
            else:
                correct_answer = entry['options-for-correct-answers'][0]
            answer = correct_answer['utterance'] + " __eou__ "
            answers.add(answer.strip())

            for i, utterance in enumerate(entry['options-for-next']):
                answer = utterance['utterance'] + " __eou__ "
                answers.add(answer.strip())

    answers = list(answers)
    with open(answers_file, "w") as vocabfile:
        for id in range(len(answers)):
            answer = answers[id]
            vocabfile.write(str(id+1) + "\t" + answer.replace("\n", "") + "\n")
    print("Saved answers to {}".format(answers_file))

    answers_return_dict = {}
    for i in range(len(answers)):
        answers_return_dict[answers[i]] = i

    return answers_return_dict


def create_test_answers_file(test_file, test_answers_file):
    answers = {}

    with open(test_file, 'rb') as f:
        json_data = ijson.items(f, 'item')
        for entry in json_data:
            for i, utterance in enumerate(entry['options-for-next']):
                answer = utterance['utterance'] + " __eou__ "
                answer_id = utterance['candidate-id']
                answers[answer_id] = answer

    answers["NONE"] = "None __eou__ "
    with open(test_answers_file, "w") as vocabfile:
        for answer_id, answer in answers.items():
            vocabfile.write(str(answer_id) + "\t" + answer.replace("\n", "") + "\n")
    print("Saved test answers to {}".format(test_answers_file))

    return answers


def get_context(dialog):
    utterances = dialog['messages-so-far']

    # Create the context
    context = ""
    speaker = None
    for msg in utterances:
        if speaker is None:
            context += msg['utterance'] + " __eou__ "
            speaker = msg['speaker']
        elif speaker != msg['speaker']:
            context += "__eot__ " + msg['utterance'] + " __eou__ "
            speaker = msg['speaker']
        else:
            context += msg['utterance'] + " __eou__ "

    context += "__eot__"
    return context

def create_train_file(train_file, train_file_out, answers):
    train_file_op = open(train_file_out, "w")
    positive_samples_count = 0
    negative_samples_count = 0

    train_data_handle = open(train_file, 'rb')
    json_data = ijson.items(train_data_handle, 'item')
    for index, entry in enumerate(json_data):
        row = str(index+1) + "\t"
        context = get_context(entry)
        row += context + "\t"

        if len(entry['options-for-correct-answers']) == 0:
            correct_answer = {}
            correct_answer['utterance'] = "None"
            target_id = "NONE"
        else:
            correct_answer = entry['options-for-correct-answers'][0]
            target_id = correct_answer['candidate-id']
        answer = correct_answer['utterance'] + " __eou__ "
        answer = answer.strip()
        correct_answer_row = row + str(answers[answer] + 1) + "\t" + "NA"
        positive_samples_count += 1
        train_file_op.write(correct_answer_row.replace("\n", "") + "\n")

        negative_answers = []
        for i, utterance in enumerate(entry['options-for-next']):
            if utterance['candidate-id'] == target_id:
                continue
            answer = utterance['utterance'] + " __eou__ "
            answer = answer.strip()
            negative_answers.append(str(answers[answer] + 1))
            negative_samples_count += 1

        if len(negative_answers) < 100:
            answer = "None __eou__"
            negative_answers.append(str(answers[answer] + 1))
            negative_samples_count += 1

        negative_answers = "|".join(negative_answers)
        negative_answer_row = row + "NA" + "\t" + negative_answers + "\t"
        train_file_op.write(negative_answer_row.replace("\n", "") + "\n")

    print("Saved training data to {}".format(train_file_out))
    print("Train - Positive samples count - {}".format(positive_samples_count))
    print("Train - Negative samples count - {}".format(negative_samples_count))
    train_file_op.close()

def create_dev_file(dev_file, dev_file_out, answers):
    dev_file_op = open(dev_file_out, "w")
    positive_samples_count = 0
    negative_samples_count = 0

    dev_data_handle = open(dev_file, 'rb')
    json_data = ijson.items(dev_data_handle, 'item')
    for index, entry in enumerate(json_data):
        row = str(index+1) + "\t"
        context = get_context(entry)
        row += context + "\t"

        if len(entry['options-for-correct-answers']) == 0:
            correct_answer = {}
            correct_answer['utterance'] = "None"
            target_id = "NONE"
        else:
            correct_answer = entry['options-for-correct-answers'][0]
            target_id = correct_answer['candidate-id']
        answer = correct_answer['utterance'] + " __eou__ "
        answer = answer.strip()
        row += str(answers[answer] + 1) + "\t"
        positive_samples_count += 1

        negative_answers = []
        for i, utterance in enumerate(entry['options-for-next']):
            if utterance['candidate-id'] == target_id:
                continue
            answer = utterance['utterance'] + " __eou__ "
            answer = answer.strip()
            negative_answers.append(str(answers[answer] + 1))
            negative_samples_count += 1

        if len(negative_answers) < 100:
            answer = "None __eou__"
            negative_answers.append(str(answers[answer] + 1))
            negative_samples_count += 1

        negative_answers = "|".join(negative_answers)
        row += negative_answers + "\t"
        dev_file_op.write(row.replace("\n", "") + "\n")

    print("Saved dev data to {}".format(dev_file_out))
    print("Dev - Positive samples count - {}".format(positive_samples_count))
    print("Dev - Negative samples count - {}".format(negative_samples_count))
    dev_file_op.close()


def create_test_file(test_file, test_file_out):
    test_file_op = open(test_file_out, "w")
    candidates_count = 0

    test_data_handle = open(test_file, 'rb')
    json_data = ijson.items(test_data_handle, 'item')
    for index, entry in enumerate(json_data):
        entry_id = entry["example-id"]
        row = str(entry_id) + "\t"
        context = get_context(entry)
        row += context + "\t"

        candidates = []
        for i, utterance in enumerate(entry['options-for-next']):
            answer_id = utterance['candidate-id']
            candidates.append(answer_id)
            candidates_count += 1

        candidates.append("NONE")

        candidates = "|".join(candidates)
        row += "NA" + "\t" + candidates + "\t"
        test_file_op.write(row.replace("\n", "") + "\n")

    print("Saved test data to {}".format(test_file_out))
    print("Test - candidates count - {}".format(candidates_count))
    test_file_op.close()


if __name__ == "__main__":
    train_file = os.path.join(FLAGS.train_in)
    dev_file = os.path.join(FLAGS.dev_in)
    #test_file = os.path.join(FLAGS.test_in)

    answers_file = os.path.join(FLAGS.answers_file)
    #test_answers_file = os.path.join(FLAGS.test_answers_file)

    train_file_out = os.path.join(FLAGS.train_out)
    dev_file_out = os.path.join(FLAGS.dev_out)
    #test_file_out = os.path.join(FLAGS.test_out)

    print("Creating vocabulary...")
    dialogs = get_dialogs(train_file)
    input_iter = iter(dialogs)
    input_iter = create_utterance_iter(input_iter)
    vocab = create_vocab(input_iter, min_frequency=FLAGS.min_word_frequency)
    print("Total vocabulary size: {}".format(len(vocab.vocabulary_)))

    # Create vocabulary txt file
    vocab_file = os.path.join(FLAGS.save_vocab_path)
    write_vocabulary(vocab, vocab_file)

    # Create answers txt file
    answers = create_answers_file(train_file, dev_file, answers_file)

    ## Once we have the test_set, we need to add test_file here too!
    #answers_test = create_test_answers_file(test_file, test_answers_file)

    # Create train txt file
    create_train_file(train_file, train_file_out, answers)
    # Create dev txt file
    create_dev_file(dev_file, dev_file_out, answers)

    ## Once we have the test_set, we need to add test_file here too!
    # Create test txt file
    #create_test_file(test_file, test_file_out)
