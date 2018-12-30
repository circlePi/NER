from preprocessing.data_processor import data_helper
from Io.data_loader import BatchIterator

from net.ner import BISLTM_CRF
from train.train import fit

import config.config as config
from util.porgress_util import ProgressBar


def bilstm_crf():
    # 数据预处理
    word2id, epoch_size = data_helper(vocab_size=config.vocab_size, max_len=config.max_len, min_freq=1,
                                      valid_size=0.2, random_state=2018, shuffle=True, is_debug=config.is_debug)

    vocab_size = len(word2id)

    # 初始化进度条
    pbar = ProgressBar(epoch_size=epoch_size, batch_size=config.batch_size)

    # 加载batch
    bi = BatchIterator(config.TRAIN_FILE,
                       config.VALID_FILE,
                       config.batch_size, fix_length=config.max_len,
                       x_var="text", y_var="label")
    train, valid = bi.create_dataset()
    train_iter, val_iter = bi.get_iterator(train, valid)

    model = BISLTM_CRF(
        vocab_size=config.vocab_size,
        word_embedding_dim=config.word_embedding_dim,
        word2id=word2id,
        hidden_size=128, bi_flag=True,
        num_layer=1, input_size=config.word_embedding_dim,
        cell_type=config.cell_type,
        dropout=config.dropout,
        num_tag=len(config.labels),
        tag2ix=config.tag_to_ix,
        checkpoint_dir=config.checkpoint_dir
    )

    # 训练
    fit(model, train_iter, val_iter,
        config.num_epoch, pbar,
        config.lr_decay_mode,
        config.initial_lr, verbose=1)