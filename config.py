# train file
train_src_file = "./squad/para-train.txt"
train_trg_file = "./squad/tgt-train.txt"
# dev file
dev_src_file = "./squad/para-dev.txt"
dev_trg_file = "./squad/tgt-dev.txt"
# test file
test_src_file = "./squad/para-test.txt"
test_trg_file = "./squad/tgt-test.txt"
# commonsense file
commonsense_file = "./data/resource.json"
entity_embedding = "./data/entity.pkl"
relation_embedding = "./data/relation.pkl"
entity_file = "./data/entity.txt"
entity_vector = "./data/entity_transE.txt"
relation_file = "./data/relation.txt"
relation_vector = "./data/relation_transE.txt"
train_csfile = "./data/paracs-train.json"
dev_csfile = "./data/paracs-dev.json"
test_csfile = "./data/paracs-test.json"

entity_size = 21471
relation_size = 44
graph_vector_size = 100

# embedding and dictionary file
embedding = "./data/embedding.pkl"
word2idx_file = "./data/word2idx.pkl"
ent2idx_file = "./data/ent2idx.pkl"
rel2idx_file = "./data/rel2idx.pkl"

model_path = "./save/model.pt"

device = "cuda:0"
use_gpu = True
debug = False
vocab_size = 45000
freeze_embedding = True

num_epochs = 30
max_len = 400
num_layers = 2
hidden_size = 300
embedding_size = 300
lr = 0.02
batch_size = 32
dropout = 0.3
max_grad_norm = 5.0

use_pointer = True
beam_size = 10
min_decode_step = 8
max_decode_step = 30
output_dir = "./result/pointer_maxout_ans"
