import argparse
import pickle
from preprocess_data import *
from crf import CRFModel
from utils import *


def main():
    # Parse the cmd arguments
    parser = argparse.ArgumentParser()

    # Path and file configs
    parser.add_argument('--data_path', default='../data/renMinRiBao', help='The dataset path.', type=str)
    parser.add_argument('--model_path', default='./model/crf.pkl', help='The model will be saved to this path.', type=str)
    parser.add_argument('--train_file', default='train_data.txt', type=str)
    parser.add_argument('--val_file', default='val_data.txt', type=str)
    parser.add_argument('--test_file', default='test_data.txt', type=str)
    parser.add_argument('--tag2id_file', default='tags.txt', type=str)

    args = parser.parse_args()

    # Every word must be mapped to a unique index
    word2id, id2word, word_cnt = build_vocab(args)
    args.vocab_size = word_cnt
    args.id2word = id2word

    tag2id = {}
    id2tag = []
    with open(os.path.join(args.data_path, args.tag2id_file), 'r') as fr:
        for i, item in enumerate(fr.readlines()):
            tag2id[item.strip('\n')] = i
            id2tag.append(item.strip('\n'))
    args.tag2id = tag2id
    args.id2tag = id2tag

    # Load the datasets
    train_word_lists, train_label_lists = preprocess(os.path.join(args.data_path, args.train_file))
    test_word_lists, test_label_lists = preprocess(os.path.join(args.data_path, args.test_file))
    print("construction finished!")

    model = CRFModel()
    model.train(train_word_lists, train_label_lists)
    print("training finished!")
    with open(args.model_path, "wb") as f:
        pickle.dump(model, f)
    pred_label_lists = model.test(test_word_lists)
    print("prediction finished!")
    metric = Metric(args.id2word, args.id2tag)
    length = len(test_word_lists)
    for i in range(length):
        metric.add(test_word_lists[i], pred_label_lists[i], test_label_lists[i])
    p, r, f1 = metric.get()
    print('Test over. P: {:5.2f} | R: {:5.2f} | F1: {:5.2f}'.format(p * 100, r * 100, f1 * 100))


if __name__ == '__main__':
    main()