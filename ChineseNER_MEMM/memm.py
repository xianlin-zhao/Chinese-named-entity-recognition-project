class MEMM(object):
    def __init__(self, config):
        self.config = config
        self.vocab_size = config.vocab_size
        self.tag2id = config.tag2id
        self.id2tag = config.id2tag
        self.id2word = config.id2word
        self.o_id2tag = []
        for tag in self.id2tag:
            self.o_id2tag.append(tag)
        self.o_id2tag.append("S")

        self.count_tag = {}  # 每个标签的数量

        self.Pi = {}  # 初始标签的分布
        self.emit = {}  # 在已知上一个状态和这一个观测的条件下，这一个标签的概率

        for tag in self.id2tag:
            self.count_tag[tag] = 0  # 初始标签归零
        for A in self.id2tag:  # 三重循环
            word_dict = {}
            for B in self.o_id2tag:
                tmp = {}
                for wordA in self.id2word:
                    tmp[wordA] = 0.
                word_dict[B] = tmp
            self.emit[A] = word_dict

    def train(self, sent_lists, label_lists):
        length = len(label_lists)
        for labels in label_lists:  # 初始标签概率
            if self.id2tag[labels[0]] in self.Pi.keys():
                self.Pi[self.id2tag[labels[0]]] += 1.0 / length
            else:
                self.Pi[self.id2tag[labels[0]]] = 1.0 / length
            self.count_tag[self.id2tag[labels[0]]] += 1  # 记录对应标签的数量
        for key in self.Pi.keys():
            if self.Pi[key] == 0.:
                self.Pi[key] = 1e-10  # 对于没出现的，赋一个很小很小的值

        for labels in label_lists:
            label_len = len(labels)
            for i in range(1, label_len):
                self.count_tag[self.id2tag[labels[i]]] += 1  # 还是计算对应标签的数量（刚才只算了每句话的第一个位置）

        for i in range(length):
            label_len = len(label_lists[i])
            for j in range(label_len):  # [i, j, k]代表：[这个标签，上一个标签，这个字]，对数据集进行统计
                if j == 0:
                    self.emit[self.id2tag[label_lists[i][j]]]["S"][self.id2word[sent_lists[i][j]]] += 1.0
                else:
                    self.emit[self.id2tag[label_lists[i][j]]][self.id2tag[label_lists[i][j - 1]]][self.id2word[sent_lists[i][j]]] += 1.0

        # 计算emit，变成概率的形式
        for tag in self.id2tag:
            for key_pre in self.emit[tag].keys():
                for key_word in self.emit[tag][key_pre].keys():
                    self.emit[tag][key_pre][key_word] = 1.0 * self.emit[tag][key_pre][key_word] / self.count_tag[tag]

        for i in self.emit.keys():
            for j in self.emit[i].keys():
                for k in self.emit[i][j].keys():
                    if self.emit[i][j][k] == 0.:
                        self.emit[i][j][k] = 1e-10

    def test(self, sent_lists):
        pred_label_lists = []
        for sent in sent_lists:
            pred_label_list = self.decoding(sent)  # 对测试集每一句话进行decoding
            pred_label_lists.append(pred_label_list)
        return pred_label_lists

    def decoding(self, sent):
        alllines = []  # 元素都是字典，存储每一层tag和对应的最大概率值
        start = {}  # 第一层（因为要特殊处理"S"）
        for tag in self.id2tag:
            if tag in self.Pi.keys():
                start[tag] = self.Pi[tag] * self.emit[tag]["S"][self.id2word[sent[0]]]
        alllines.append(start)

        length = len(sent)
        path = []  # 相当于标记函数，记录上一个tag
        pro = "O"
        for i in range(1, length):
            next_dict = {}  # 记录概率
            new_path = {}  # 记录上一个tag
            for tag in self.id2tag:
                now_max = 0
                for key in alllines[i - 1].keys():  # 枚举上一个标签
                    value = alllines[i - 1][key] * self.emit[tag][key][self.id2word[sent[i]]]
                    if value > now_max:  # 随时更新最大概率值以及对应的上一个标签
                        now_max = value
                        pro = key
                next_dict[tag] = now_max
                new_path[tag] = pro
            path.append(new_path)
            alllines.append(next_dict)

        now_max = 0
        end = "O"
        for key in alllines[-1].keys():  # 最后一层的最大概率的标签
            if alllines[-1][key] > now_max:
                end = key

        result = [end]
        for i in range(len(alllines) - 2, -1, -1):  # 倒着找最优路径
            for key in path[i].keys():
                if key == result[len(alllines) - i - 2]:
                    result.append(path[i][key])
        result.reverse()
        for i in range(len(result)):
            result[i] = self.tag2id[result[i]]  # 把标签名字转换成id
        return result
