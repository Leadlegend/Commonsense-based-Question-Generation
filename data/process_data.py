import sys
import json

sys.path.insert(0, '../')

import config
from data_utils import (make_conll_format, make_embedding, make_vocab,
                        make_vocab_from_squad, process_file, make_graph_vector)


def make_sent_dataset():
    train_src_file = "./para-train.txt"
    train_trg_file = "./tgt-train.txt"

    embedding_file = "./glove.840B.300d.txt"
    embedding = "./embedding.pkl"
    word2idx_file = "./word2idx.pkl"
    # make vocab file
    word2idx = make_vocab(train_src_file, train_trg_file, word2idx_file, config.vocab_size)
    make_embedding(embedding_file, embedding, word2idx)


def make_para_dataset():
    embedding_file = "./glove.840B.300d.txt"
    embedding = "./embedding.pkl"
    src_word2idx_file = "./word2idx.pkl"
    ent2idx_file = "./ent2idx.pkl"
    rel2idx_file = "./rel2idx.pkl"
    entity_embedding = "./entity.pkl"
    relation_embedding = "./relation.pkl"

    train_squad = "../squad/train-v1.1.json"
    dev_squad = "../squad/dev-v1.1.json"

    train_src_file = "../squad/para-train.txt"
    train_trg_file = "../squad/tgt-train.txt"
    train_cs_file = "./paracs-train.json"
    dev_src_file = "../squad/para-dev.txt"
    dev_trg_file = "../squad/tgt-dev.txt"
    dev_cs_file = "./paracs-dev.json"

    test_src_file = "../squad/para-test.txt"
    test_trg_file = "../squad/tgt-test.txt"
    test_cs_file = "./paracs-test.json"
    ent_vector = "./entity_transE.txt"
    rel_vector = "./relation_transE.txt"
    ent_file = "./entity.txt"
    rel_file = "./relation.txt"
    cs_file = "./resource.json"

    database = dict()
    with open(cs_file, "r") as f:
        d = json.load(f)
        if d["dict_csk"] is not None:
            database = d["dict_csk"]

    # process the graph vector through the static attention mechanism
    _, _, ent2idx, rel2idx = make_graph_vector(entity_embedding,
                                               relation_embedding,
                                               ent_vector,
                                               ent_file,
                                               rel_vector,
                                               rel_file,
                                               ent2idx_file,
                                               rel2idx_file
                                               )
    # pre-process training data
    train_examples, counter, num = process_file(train_squad, ent2idx, rel2idx, database)
    make_conll_format(train_examples, train_src_file, train_trg_file, train_cs_file, num)
    word2idx = make_vocab_from_squad(src_word2idx_file, counter, config.vocab_size)
    make_embedding(embedding_file, embedding, word2idx)

    # split dev into dev and test
    dev_test_examples, _, num = process_file(dev_squad, ent2idx, rel2idx, database)
    # random.shuffle(dev_test_examples)
    num_dev = len(dev_test_examples) // 2
    dev_examples = dev_test_examples[:num_dev]
    test_examples = dev_test_examples[num_dev:]
    make_conll_format(dev_examples, dev_src_file, dev_trg_file, dev_cs_file, num)
    make_conll_format(test_examples, test_src_file, test_trg_file, test_cs_file, num)


if __name__ == "__main__":
    # make_sent_dataset()
    make_para_dataset()
