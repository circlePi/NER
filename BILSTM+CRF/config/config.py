ROOT_DIR = '/home/daizelin/NER'
TRAIN_FILE = 'output/intermediate/train.json'
VALID_FILE = 'output/intermediate/valid.json'
RAW_SOURCE_DATA = 'data/source_BIO_2014_cropus.txt'
RAW_TARGET_DATA = 'data/target_BIO_2014_cropus.txt'

WORD2ID_FILE = 'output/intermediate/word2id.pkl'
EMBEDDING_FILE = 'embedding/peopel_paper_min_count_1_window_5_300d.word2vec'
LOG_PATH = 'output/logs'

checkpoint_dir = 'output/checkpoints/bilstm_ner.ckpt'
plot_path = 'output/images/img'


# -----------PARAMETERS----------------
tag_to_ix = {
    "B_PER": 0,   # 人名
    "I_PER": 1,
    "B_LOC": 2,   # 地点
    "I_LOC": 3,
    "B_ORG": 4,   # 机构
    "I_ORG": 5,
    "B_T": 6,     # 时间
    "I_T": 7,
    "O": 8,       # 其他
    "SOS": 9,     # 起始符
    "EOS":10      # 结束符
}

labels = [i for i in range(0, 9)]

flag_words = ['<pad>', '<unk>']
max_len = 100
vocab_size = 10000
is_debug = False

# ------------NET　PARAMS----------------
use_mem_track = False
device = 0
use_cuda = True
word_embedding_dim = 300
batch_size = 128
cell_type ='GRU'
dropout = 0.5
num_epoch = 4
lr_decay_mode = 'custom_decay'
initial_lr = 0.001