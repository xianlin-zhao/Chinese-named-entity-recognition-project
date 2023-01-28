from sklearn_crfsuite import CRF


def word2features(sent, i):
    word = sent[i]
    if i == 0:
        prev_word = "<s>"
    else:
        prev_word = sent[i - 1]
    if i == len(sent) - 1:
        next_word = "<s/>"
    else:
        next_word = sent[i + 1]
    features = {
        'w': word,
        'w-1': prev_word,
        'w+1': next_word,
        'w-1:w': prev_word+word,
        'w:w+1': word+next_word,
        'bias': 1
    }
    return features


def sent2features(sent):
    ans = [word2features(sent, i) for i in range(len(sent))]
    return ans


class CRFModel(object):
    def __init__(self,
                 c1=0.1,
                 c2=0.07,
                 max_iterations=80,
                 all_possible_transitions=False
                 ):
        self.model = CRF(
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)

    def train(self, sent_lists, label_lists):
        features = [sent2features(s) for s in sent_lists]
        self.model.fit(features, label_lists)

    def test(self, sent_lists):
        features = [sent2features(s) for s in sent_lists]
        pred_label_lists = self.model.predict(features)
        return pred_label_lists
