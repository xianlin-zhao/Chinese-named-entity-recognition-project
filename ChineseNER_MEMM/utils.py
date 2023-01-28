# Compute the precision, recall and F1 score
class Metric:
    def __init__(self, id2word, id2tag):
        self.gold_num = 0
        self.pred_num = 0
        self.correct = 0
        self.id2word = id2word
        self.id2tag = id2tag

    def add(self, sent, pred, gold):
        pred_entity = self.get_entity(sent, pred)
        gold_entity = self.get_entity(sent, gold)
        self.gold_num += len(gold_entity)
        self.pred_num += len(pred_entity)
        self.correct += len([item for item in pred_entity if item in gold_entity])

    def get_entity(self, sent, tag):
        res = []
        entity = []
        for j in range(len(sent)):
            if sent[j] == 0 or tag[j] == 0:
                continue
            if self.id2tag[tag[j]][0] == 'B':
                entity = [self.id2word[sent[j]] + '/' + self.id2tag[tag[j]]]
            elif self.id2tag[tag[j]][0] == 'I' and len(entity) != 0 \
                    and entity[-1].split('/')[1][1:] == self.id2tag[tag[j]][1:]:
                entity.append(self.id2word[sent[j]] + '/' + self.id2tag[tag[j]])
                if j == len(sent) - 1 or self.id2tag[tag[j + 1]][0] == 'O':
                    entity.append(str(j))
                    res.append(entity)
                    entity = []
            else:
                entity = []
        return res

    def get(self):
        if self.pred_num == 0 or self.gold_num == 0:
            return 0, 0, 0
        p = self.correct / self.pred_num
        r = self.correct / self.gold_num
        if p + r == 0:
            return 0, 0, 0
        f1 = 2*p*r / (p + r)
        return p, r, f1