import torch


class HMM(object):
    def __init__(self, N, M):
        self.N = N  # 状态数，即tag的种类
        self.M = M  # 观测数，即有多少不同的字
        self.A = torch.zeros(N, N)  # 状态转移概率矩阵
        self.B = torch.zeros(N, M)  # 观测概率矩阵
        self.Pi = torch.zeros(N)  # 初始状态概率

    def train(self, sent_lists, label_lists):
        for labels in label_lists:
            length = len(labels)
            for i in range(length - 1):
                self.A[labels[i]][labels[i + 1]] += 1  # 标签之间的转移（状态转移）
        self.A[self.A == 0.] = 1e-10
        self.A = self.A / self.A.sum(dim=1, keepdim=True)

        for labels, sent in zip(label_lists, sent_lists):
            for label, word in zip(labels, sent):
                self.B[label][word] += 1  # 标签对应的字（状态对应的观测）
        self.B[self.B == 0.] = 1e-10
        self.B = self.B / self.B.sum(dim=1, keepdim=True)

        for labels in label_lists:
            self.Pi[labels[0]] += 1  # 起点的标签（初始状态）
        self.Pi[self.Pi == 0.] = 1e-10
        self.Pi = self.Pi / self.Pi.sum()

    def test(self, sent_lists, word2id, tag2id):
        pred_label_lists = []
        for sent in sent_lists:
            pred_label_list = self.decoding(sent, word2id, tag2id)  # 对于每句话用viterbi解码得到概率最大的路径（标签序列）
            pred_label_lists.append(pred_label_list)
        return pred_label_lists

    def decoding(self, sent, word2id, tag2id):  # 用动态规划求概率最大路径
        A = torch.log(self.A)
        B = torch.log(self.B)
        Pi = torch.log(self.Pi)  # 把概率变成对数，则相乘变成相加，也防止下溢

        length = len(sent)
        # [i, j]表示序列的第j个字标注为i的所有单个序列出现的概率最大值
        viterbi = torch.zeros(self.N, length)  # 标签数 * 序列长度
        backpointer = torch.zeros(self.N, length).long()  # 记录序列第j个字标签为i时，上一个的标签

        Bt = B.t()
        if sent[0] == word2id['UNK']:  # 如果第一个字不在字典里，就假设状态的概率分布是均匀的
            bt = torch.log(torch.ones(self.N) / self.N)
        else:
            bt = Bt[sent[0]]
        viterbi[:, 0] = Pi + bt  # 概率相乘，log就是相加
        backpointer[:, 0] = -1

        for step in range(1, length):
            # bt是这个字的标签概率分布
            if sent[step] == word2id['UNK']:
                bt = torch.log(torch.ones(self.N) / self.N)
            else:
                bt = Bt[sent[step]]
            for tag_id in range(len(tag2id)):
                max_prob, max_id = torch.max(viterbi[:, step - 1] + A[:, tag_id], dim=0)  # 转移
                viterbi[tag_id, step] = max_prob + bt[tag_id]  # 观测
                backpointer[tag_id, step] = max_id  # 标记函数

        best_path_prob, best_path_pointer = torch.max(viterbi[:, length - 1], dim=0)
        best_path_pointer = best_path_pointer.item()
        best_path = []
        best_path.append(best_path_pointer)
        for back_step in range(length - 1, 0, -1):
            best_path_pointer = backpointer[best_path_pointer, back_step]  # 倒着不断找前一个标签
            best_path_pointer = best_path_pointer.item()
            best_path.append(best_path_pointer)
        best_path.reverse()  # 倒着找完，再把列表翻转过来
        return best_path
