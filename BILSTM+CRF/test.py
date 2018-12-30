from preprocessing.data_processor import data_helper
from Io.data_loader import BatchIterator

import config.config as config

def run():
    print('start...')
    data_helper(10000, 25, is_debug=False)

    bi = BatchIterator(config.TRAIN_FILE, config.VALID_FILE, config.batch_size, config.max_len)
    train, valid = bi.create_dataset()
    train_iter, valid_iter = bi.get_iterator(train, valid)
    batch = next(iter(train_iter))
    print(batch)

if __name__ == '__main__':
    run()