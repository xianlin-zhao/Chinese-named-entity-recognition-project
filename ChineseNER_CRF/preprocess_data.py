import os


def build_vocab(config):
    word2id = {'PAD': 0}
    word_cnt = 1
    id2word = ['PAD']
    with open(os.path.join(config.data_path, config.train_file), 'r', encoding='utf-8') as fr:
        raw = fr.readlines()
    for item in raw:
        if item[0] == '\n':
            continue
        t = item[0]
        if t not in word2id:
            word2id[t] = word_cnt
            word_cnt += 1
            id2word.append(t)
    word2id['UNK'] = word_cnt
    word_cnt += 1
    id2word.append('UNK')
    assert len(word2id) == len(id2word)
    return word2id, id2word, word_cnt


def preprocess(data_path):
    with open(data_path, 'r', encoding='utf-8') as fr:
        raw = fr.readlines()
    x = []
    y = []

    xx, yy = [], []
    for item in raw:
        if item[0] == '\n':
            if len(xx) == 0:
                continue
            assert len(xx) == len(yy)
            x.append(xx)
            y.append(yy)
            xx, yy = [], []
            continue
        item = item.strip('\n')
        xx.append(item[0])
        yy.append(item[2:])
    return x, y
