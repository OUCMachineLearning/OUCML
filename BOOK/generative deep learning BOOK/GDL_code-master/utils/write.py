#!/usr/bin/env python

from collections import Counter

import csv

import random

import numpy as np


_MAX_BATCH_SIZE = 128
_MAX_DOC_LENGTH = 200

PADDING_WORD = "<PAD>"
UNKNOWN_WORD = "<UNK>"
START_WORD = "<START>"
END_WORD = "<END>"

_word_to_idx = {}
_idx_to_word = []


def _add_word(word):
    idx = len(_idx_to_word)
    _word_to_idx[word] = idx
    _idx_to_word.append(word)
    return idx


PADDING_TOKEN = _add_word(PADDING_WORD)
UNKNOWN_TOKEN = _add_word(UNKNOWN_WORD)
START_TOKEN = _add_word(START_WORD)
END_TOKEN = _add_word(END_WORD)


embeddings_path = './data/glove/glove.6B.100d.trimmed.txt'

with open(embeddings_path) as f:
    line = f.readline()
    chunks = line.split(" ")
    dimensions = len(chunks) - 1
    f.seek(0)

    vocab_size = sum(1 for line in f)
    vocab_size += 4 #3 
    f.seek(0)

    glove = np.ndarray((vocab_size, dimensions), dtype=np.float32)
    glove[PADDING_TOKEN] = np.random.normal(0, 0.02, dimensions)
    glove[UNKNOWN_TOKEN] = np.random.normal(0, 0.02, dimensions)
    glove[START_TOKEN] = np.random.normal(0, 0.02, dimensions)
    glove[END_TOKEN] = np.random.normal(0, 0.02, dimensions)

    for line in f:
        chunks = line.split(" ")
        idx = _add_word(chunks[0])
        glove[idx] = [float(chunk) for chunk in chunks[1:]]
        if len(_idx_to_word) >= vocab_size:
            break




def look_up_word(word):
    return _word_to_idx.get(word, UNKNOWN_TOKEN)


def look_up_token(token):
    return _idx_to_word[token]



def _tokenize(string):
    return [word.lower() for word in string.split(" ")]


def _prepare_batch(batch):
    id_to_indices = {}
    document_ids = []
    document_text = []
    document_words = []
    answer_text = []
    answer_indices = []
    question_text = []
    question_input_words = []
    question_output_words = []
    for i, entry in enumerate(batch):
        id_to_indices.setdefault(entry["document_id"], []).append(i)
        document_ids.append(entry["document_id"])
        document_text.append(entry["document_text"])
        document_words.append(entry["document_words"])
        answer_text.append(entry["answer_text"])
        answer_indices.append(entry["answer_indices"])
        question_text.append(entry["question_text"])

        question_words = entry["question_words"]
        question_input_words.append([START_WORD] + question_words)
        question_output_words.append(question_words + [END_WORD])

    batch_size = len(batch)
    max_document_len = max((len(document) for document in document_words), default=0)
    max_answer_len = max((len(answer) for answer in answer_indices), default=0)
    max_question_len = max((len(question) for question in question_input_words), default=0)

    document_tokens = np.zeros((batch_size, max_document_len), dtype=np.int32)
    document_lengths = np.zeros(batch_size, dtype=np.int32)
    answer_labels = np.zeros((batch_size, max_document_len), dtype=np.int32)
    answer_masks = np.zeros((batch_size, max_answer_len, max_document_len), dtype=np.int32)
    answer_lengths = np.zeros(batch_size, dtype=np.int32)
    question_input_tokens = np.zeros((batch_size, max_question_len), dtype=np.int32)
    question_output_tokens = np.zeros((batch_size, max_question_len), dtype=np.int32)
    question_lengths = np.zeros(batch_size, dtype=np.int32)

    for i in range(batch_size):
        for j, word in enumerate(document_words[i]):
            document_tokens[i, j] = look_up_word(word)
        document_lengths[i] = len(document_words[i])

        for j, index in enumerate(answer_indices[i]):
            for shared_i in id_to_indices[batch[i]["document_id"]]:
                answer_labels[shared_i, index] = 1
            answer_masks[i, j, index] = 1
        answer_lengths[i] = len(answer_indices[i])

        for j, word in enumerate(question_input_words[i]):
            question_input_tokens[i, j] = look_up_word(word)
        for j, word in enumerate(question_output_words[i]):
            question_output_tokens[i, j] = look_up_word(word)
        question_lengths[i] = len(question_input_words[i])

    return {
        "size": batch_size,
        "document_ids": document_ids,
        "document_text": document_text,
        "document_words": document_words,
        "document_tokens": document_tokens,
        "document_lengths": document_lengths,
        "answer_text": answer_text,
        "answer_indices": answer_indices,
        "answer_labels": answer_labels,
        "answer_masks": answer_masks,
        "answer_lengths": answer_lengths,
        "question_text": question_text,
        "question_input_tokens": question_input_tokens,
        "question_output_tokens": question_output_tokens,
        "question_lengths": question_lengths,
    }


def collapse_documents(batch):
    seen_ids = set()
    keep = []

    for i in range(batch["size"]):
        id = batch["document_ids"][i]
        if id in seen_ids:
            continue

        keep.append(i)
        seen_ids.add(id)

    result = {}
    for key, value in batch.items():
        if key == "size":
            result[key] = len(keep)
        elif isinstance(value, np.ndarray):
            result[key] = value[keep]
        else:
            result[key] = [value[i] for i in keep]
    return result


def expand_answers(batch, answers):
    new_batch = []

    for i in range(batch["size"]):
        split_answers = []
        last = None
        for j, tag in enumerate(answers[i]):
            if tag:
                if last != j - 1:
                    split_answers.append([])
                split_answers[-1].append(j)
                last = j

        if len(split_answers) > 0:

            answer_indices = split_answers[0]
        # for answer_indices in split_answers:
            document_id = batch["document_ids"][i]
            document_text = batch["document_text"][i]
            document_words = batch["document_words"][i]
            answer_text = " ".join(document_words[i] for i in answer_indices)
            new_batch.append({
                "document_id": document_id,
                "document_text": document_text,
                "document_words": document_words,
                "answer_text": answer_text,
                "answer_indices": answer_indices,
                "question_text": "",
                "question_words": [],
            })
        else:
            new_batch.append({
                "document_id": batch["document_ids"][i],
                "document_text": batch["document_text"][i],
                "document_words": batch["document_words"][i],
                "answer_text": "",
                "answer_indices": [],
                "question_text": "",
                "question_words": [],
            })

    return _prepare_batch(new_batch)


def _read_data(path):
    stories = {}

    with open(path) as f:
        header_seen = False
        for row in csv.reader(f):
            if not header_seen:
                header_seen = True
                continue

            document_id = row[0]

            existing_stories = stories.setdefault(document_id, [])

            document_text = row[1]
            if existing_stories and document_text == existing_stories[0]["document_text"]:
                # Save memory by sharing identical documents
                document_text = existing_stories[0]["document_text"]
                document_words = existing_stories[0]["document_words"]
            else:
                document_words = _tokenize(document_text)
                document_words = document_words[:_MAX_DOC_LENGTH]

            question_text = row[2]
            question_words = _tokenize(question_text)

            answer = row[3]
            answer_indices = []
            for chunk in answer.split(","):
                start, end = (int(index) for index in chunk.split(":"))
                if end < _MAX_DOC_LENGTH:
                    answer_indices.extend(range(start, end))
            answer_text = " ".join(document_words[i] for i in answer_indices)

            if len(answer_indices) > 0:
                existing_stories.append({
                    "document_id": document_id,
                    "document_text": document_text,
                    "document_words": document_words,
                    "answer_text": answer_text,
                    "answer_indices": answer_indices,
                    "question_text": question_text,
                    "question_words": question_words,
                })

     

    return stories


def _process_stories(stories):
    batch = []
    vals = list(stories.values())
    random.shuffle(vals)

    for story in vals:
        if len(batch) + len(story) > _MAX_BATCH_SIZE:
            yield _prepare_batch(batch)
            batch = []
        batch.extend(story)

    if batch:
        yield _prepare_batch(batch)


_training_stories = None
_test_stories = None

def _load_training_stories():
    global _training_stories
    _training_stories = _read_data("./data/qa/train.csv")
    return _training_stories

def _load_test_stories():
    global _test_stories
    _test_stories = _read_data("./data/qa_test/my_test.csv")
    return _test_stories

def training_data():
    return _process_stories(_load_training_stories())

def test_data():
    return _process_stories(_load_test_stories())


def trim_embeddings():
    document_counts = Counter()
    question_counts = Counter()
    for data in [_load_training_stories().values(), _load_test_stories().values()]:
        
        for stories in data:

            if len(stories) > 0:
                document_counts.update(stories[0]["document_words"])
                for story in stories:
                    question_counts.update(story["question_words"])

    keep = set()
    for word, count in question_counts.most_common(5000):
        keep.add(word)
    for word, count in document_counts.most_common():
        if len(keep) >= 10000:
            break
        keep.add(word)

    with open("./data/glove/glove.6B.100d.txt") as f:
        with open("./data/glove/glove.6B.100d.trimmed.txt", "w") as f2:
            for line in f:
                if line.split(" ")[0] in keep:
                    f2.write(line)


if __name__ == '__main__':
    trim_embeddings()
