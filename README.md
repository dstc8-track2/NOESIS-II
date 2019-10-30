# Dialog System Technology Challenges 8 (DSTC 8) - Track 2 
## NOESIS II: Predicting Responses, Identifying Success, and Managing Complexity in Task-Oriented Dialogue

### Introduction ###
Building on the success of [DSTC 7 Track 1](https://ibm.github.io/dstc-noesis/public/index.html) (NOESIS: Noetic End-to-End Response Selection Challenge), we propose an extension of the task, incorporating new elements that are vital for the creation of a deployed task-oriented dialogue system. Specifically, we add three new dimensions to the challenge: 
1.  Conversations with more than 2 participants
2.  Predicting whether a dialogue has solved the problem yet,
3.  Handling multiple simultaneous conversations. 
Each of these adds an exciting new dimension and brings the task closer to the creation of systems able to handle the complexity of real-world conversation.

This challenge is offered with two goal oriented dialog datasets, used in 4 subtasks. A participant may participate in one, several, or all the subtasks. A full description of the track is available [here](https://drive.google.com/file/d/1rCCRsuZ7rq2KGEnT-pBCF6WS47yEsIJA/view).

**Please visit this webpage often to remain updated about baseline results and more material.**

### Ubuntu dataset ###
A new set of disentangled Ubuntu IRC dialogs will be provided in this challenge. The dataset consists of multi party conversations extracted from the Ubuntu IRC channel. A typical dialog starts with a question that was asked by participant_1, and then other partipants responds with either an answer or follow-up questions that then lead to a back-and-forth conversation. In this challenge, the context of each dialog contains more than 3 turns which occurred between the participants and the next turn in the conversation should be selected from the given set of candidate utterances. Relevant external information of the form of Linux manual pages and Ubuntu discussion forums is also provided.

If you use the Ubuntu data, please also cite the paper in which we describe its creation:
```
@InProceedings{acl19disentangle,
  author    = {Jonathan K. Kummerfeld and Sai R. Gouravajhala and Joseph Peper and Vignesh Athreya and Chulaka Gunasekara and Jatin Ganhotra and Siva Sankalp Patel and Lazaros Polymenakos and Walter S. Lasecki},
  title     = {A Large-Scale Corpus for Conversation Disentanglement},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {July},
  year      = {2019},
}
```

### Advising dataset ###
This dataset contains two party dialogs that simulate a discussion between a student and an academic advisor. The purpose of the dialogs is to guide the student to pick courses that fit not only their curriculum, but also personal preferences about time, difficulty, areas of interest, etc. These conversations were collected by having students at the University of Michigan act as the two roles using provided personas. Structured information in the form of a database of course information will be provided, as well as the personas (though at test time only information available to the advisor will be provided, i.e. not the explicit student preferences). The data also includes paraphrases of the sentences and of the target responses.

Note: the Advising data is considerably smaller than the Ubuntu data, but more focused in topic.

### Sub-tasks ###
We are considered several subtasks that have similar structure, but vary in the output space and available context. In the table below, [x] indicates that the subtask is evaluated on the marked dataset.

|Subtask number|  Description | Ubuntu  | Advising   |
|--------------| ------------ | ------------ | ------------ |
|1|Given the disentangled conversation, select the next utterance from a candidate pool of 100 which might not contain the correct next utterance |  [x] |  [x]  |
|2|Given a section of the IRC channel, select the next utterance from a candidate pool of 100 which might not contain the correct next utterance |  [x]  |   |
|3|Given a conversation, predict where in the conversation the problem is solved (if at all).||[x]  |
|4|Given a section of the IRC channel, identify a set of conversations contained within that section|  [x] ||


### Data ###
The datasets are available from the following links. 

- Training and dev data - [http://ibm.biz/dstc8_track2_data](http://ibm.biz/dstc8_track2_data)
- Test data - [http://ibm.biz/dstc8_track2_test_data](http://ibm.biz/dstc8_track2_test_data)
- Ground truth of the test data - [http://ibm.biz/dstc8_test_ground_truth](http://ibm.biz/dstc8_test_ground_truth)

In addition to the training and validation dialog datasets, and extra dataset which includes paraphrases for utterances in advising dataset is also provided. 

The candidate sets that are provided for some dialogs in **subtask 1 and 2** does not include the correct next utterance.
The contestants are expected to train their models in a way that during testing they can identify such cases.

Additional external information which will be important for dialog modeling will be provided. For Ubuntu dataset, this external information comes in the form of Linux manual pages and Ubuntu discussion forums and for Advising dataset, extra information about courses will be given. 
The contestants can use the provided knowledge sources as is, or transform them to appropriate representations (e.g. knowledge graphs, continuous embeddings, etc.) that can be integrated with end-to-end dialog systems to improve accuracy.

### Data format ####

#### Subtask 1, 2 and 3 ####
Each dialog contains in training, validation and test datasets follows the JSON format which is similar to the below example.
```
{
    "data-split": "train",
        "example-id": 0,
        "messages-so-far": [
            {
                "date": "2007-02-13",
                "speaker": "participant_0",
                "time": "07:31",
                "utterance": "hi guys, i need some urgent help. i \"rm -rf'd\" a direcotry. any way i can recover it?"
            },
            {
                "date": "2007-02-13",
                "speaker": "participant_1",
                "time": "07:31",
                "utterance": "participant_0 : in short, no."
            },
            {
                "date": "2007-02-13",
                "speaker": "participant_0",
                "time": "07:31",
                "utterance": "participant_1 , are you sure?"
            },
            ...
        ],
        "options-for-correct-answers": [
            {
                "candidate-id": "3d06877cb2f0c1861b248860fa60ce07",
                "speaker": "participant_1",
                "utterance": "\"Are you sure?\" is something rm -rf never asks.."
            }
        ],
        "options-for-next": [
            {
                "candidate-id": "ace962b708d559fc462b7fdd9b6fc093",
                "speaker": "participant_1",
                "utterance": "(and if hardware is detected correctly, of course)"
            },
            {
                "candidate-id": "349efca9c3d5986a87d95fb90c1b7c04",
                "speaker": "participant_2",
                "utterance": "how do i do a simulated reboot"
            },
            ...
         ],
      "scenario": 1
}
```
The field `messages-so-far` contains the context of the dialog and `options-for-next` contains the candidates to select the next utterance from. The correct next utterance is given in the field `options-for-correct-answers`. The field `scenario` refers to the subtask.

For each dialog in advising dataset, we provide a profile that contains information used during the creation of the dialog. It has the following fields:

- Aggregated - contains student preferences, with each field matching up with a field in the course information file.
- Courses - contains two lists, first is a list of courses this student has taken (“Prior”) and second is a list of suggestions that the advisor had access to (“Suggested”).
- Term - specifies the simulated year and semester for the conversation
- Standing - specifies how far through their degree the student is.

For subtask 3, the information regarding the success of a conversation is given in the field similar to the following where the `label` indicates whether the conversation is a success or not and `position` indicates the utterance where the conversation is accepted.
```
"success-labels": [
            {
                "label": "Accept",
                "position": 8
            }
        ]
```

The label can be 'Accept', 'Reject', or 'No Decision Yet'.
There can be multiple labels, to cover cases where the student accepts/rejects multiple advisor suggestions.
If the value is 'No Decision Yet' then the position will be -1.
'No Decision Yet' appears in the task-1.advising.train.json file, and the dev and test files, but not in the task-1.advising.train.complete.json file (which contains complete conversations rather than partial ones).

#### Subtask 4 ####
This data is not in the same format as the other subtasks. There are train and dev folders containing a set of files like this:

- DATE.raw.txt
- DATE.tok.txt
- DATE.ascii.txt
- DATE.annotation.txt

The `.raw.txt` file is a sample from the IRC log from that day. The `.ascii.txt` file is a version of the raw file that we have converted to ascii. The `.tok.txt` file has the same data agian, but with automatic tokenisation and replacement of rare words with a placeholder symbol. The `.annotation.txt` file contains a series of lines, each describing a link between two messages. For example: `1002 1003 -`. This indicates that message `1002` in the logs should be linked to message `1003`. 

Note:
- Messages are counted starting at 0 and each one is a single line in the logs.
- System messages (e.g. “=== blah has joined #ubuntu”) are counted and annotated.
- A message can be linked to multiple messages both before it and after it. Each link is given separately.
- There are no links where both values are less than 1000. In other words, the annotations specify what each message is a response to, starting from message 1,000.

### Baselines ###
Baselines for subtask 1 and 4 are available [here](https://github.com/dstc8-track2/NOESIS-II/tree/baseline). 

### Evaluation ###

For subtask 1 and 2, we will expect you to return a set of 100 candidates and a probability distribution over those 100 choices. As competition metrics we will compute range of scores, including recall@k, MRR(mean reciprocal rank). The
final metric will be the average of MRR and recall@10. 

For subtask 3, the participants are expected to return the success or a failiure of the conversation and utterance in which the success is indicated. The participants will be evaluated by the accuracy of identifying success and failure. 

The participants of the subtask 4 will be evaluated based on Precision, recall, and F-
score over complete threads and several clustering metrics (Variation of Information, Adjusted Rand Index, and Adjusted Mutual Information).


### Submission ###

Your submissions should be emailed to `chulaka.gunasekara@ibm.com`, with the subject line `DSTC8_Track2_Submission`. The results should be submitted from an email address that is registered for Track 2.

You need to submit a single zipped directory containing the result files for each of the subtasks that you need to be evaluated on. The files should be named `<dataset>_subtask_<subtask_number>.json` for subtasks 1-3 and `Ubuntu_subtask_4.txt` for subtask 4. 

For subtasks 1-3, the `<dataset>` should be replaced by either `Ubuntu` or `Advising`, and the `<subtask_number>` should be replaced by the subtask number(1-3). For example, the results file for subtask 1 on Ubuntu dataset should be named as `Ubuntu_subtask_1.json`

Each results file for subtask 1 and 2 should follow the following json format.
```
[
    {
        "example-id": xxxxxxx,
        "candidate-ranking":[
            {
                "candidate-id": aaaaaa,
                "confidence": b.bbb
            },
            {
                "candidate-id": cccccc,
                "confidence": d.ddd
            },
            ...
        ]
    },
    ...
]
```
The value for the field `example-id` should contain the corresponding example-id of the test dataset. The candidate ranking field should ONLY include 100 candidates in the order of confidence.

When the correct candidate is not available in the candidate set, return `"candidate-id": "NONE"` with the confidence score as an item in the candidate-ranking list.

The results file for subtask 3 should follow the following format. 
```
[
    {
        "example-id": xxxxxxx,
        "success-labels": [
            {
                "label": "label",
                "position": N
            }
        ]
     },
     ...
]
```
The value for the field `example-id` should contain the corresponding `example-id` of the test dataset. The `label` can be 'Accept', 'Reject', or 'No Decision Yet'. The `position` should be and integer which points to the utterance where the conversation is accepted or rejected. The If the value is 'No Decision Yet' then the position should be -1. 


The results for subtask 4 should be a submitted by a single file `(Ubuntu_subtask_4.txt)` in the following format:

```
DATE: A B C...
DATE: Q W E R T Y...
...
```

Where each line describes one cluster for that date (where the date comes from the test input filename). For example, you could have:

```
2005-07-06 1000 1001 1002 1003
2005-07-06 1004 1006
2005-07-06 1005
2005-07-06 1007 1008
...
2014-06-18 1000 1002
2014-06-18 1001 1003 1004
...
```

Note, we will ignore all numbers below 1000 and if any numbers 1000 or above are missing we will assume each of those lines forms its own conversation.



### Timeline ###

- Development Phase: Jun 17 - Sep 22, 2019 (14 weeks)
- Test data out on: Sep 23, 2019
- Evaluation Phase: Sep 23 - Oct 13, 2019 (2 weeks)
- Submission deadline: Oct 13, 2019 at 11:59pm UTC-10 (midnight Hawaii)
- Announcement of the results: Oct 20, 2019
- Paper Submission: TBA
- DSTC8 Workshop: TBA

### Organizers ###

[Chulaka Gunasekara](https://researcher.watson.ibm.com/researcher/view.php?person=ibm-chulaka.gunasekara), [Luis Lastras](https://researcher.watson.ibm.com/researcher/view.php?person=us-lastrasl)  – IBM Research AI <br>
[Jonathan K. Kummerfeld](http://www.jkk.name), [Walter Lasecki](https://web.eecs.umich.edu/~wlasecki/)  – University of Michigan








