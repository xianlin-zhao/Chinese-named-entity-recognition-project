import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import math
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout, max_len=5000):
    #d_model=512,dropout=0.1,
    #max_len=5000代表事先准备好长度为5000的序列的位置编码，其实没必要，
    #一般100或者200足够了。
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model)
    #(5000,512)矩阵，保持每个位置的位置编码，一共5000个位置，
    #每个位置用一个512维度向量来表示其位置编码
    position = torch.arange(0, max_len).unsqueeze(1)
    # (5000) -> (5000,1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
      -(math.log(10000.0) / d_model))
      # (0,2,…, 4998)一共准备2500个值，供sin, cos调用
    pe[:, 0::2] = torch.sin(position * div_term) # 偶数下标的位置
    pe[:, 1::2] = torch.cos(position * div_term) # 奇数下标的位置
    pe = pe.unsqueeze(0)
    # (5000, 512) -> (1, 5000, 512) 为batch.size留出位置
    self.register_buffer('pe', pe)
  def forward(self, x):
    x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
    # 接受1.Embeddings的词嵌入结果x，
    #然后把自己的位置编码pe，封装成torch的Variable(不需要梯度)，加上去。
    #例如，假设x是(30,10,512)的一个tensor，
    #30是batch.size, 10是该batch的序列长度, 512是每个词的词嵌入向量；
    #则该行代码的第二项是(1, min(10, 5000), 512)=(1,10,512)，
    #在具体相加的时候，会扩展(1,10,512)为(30,10,512)，
    #保证一个batch中的30个序列，都使用（叠加）一样的位置编码。
    return self.dropout(x) # 增加一次dropout操作

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.device = config.device

        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.tag2id = config.tag2id
        self.Position = PositionalEncoding(self.emb_dim, 0.1)
        self.k=4
        self.hiddenatt_dim=self.hidden_dim//4
        self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.ffn=nn.ReLU(inplace=False)
        self.ffnlinear=nn.Linear(self.hidden_dim,self.hidden_dim)  #前馈神经网络
        self.i2q1=nn.Linear(self.emb_dim,self.hiddenatt_dim)
        self.i2k1=nn.Linear(self.emb_dim,self.hiddenatt_dim)
        self.i2v1=nn.Linear(self.emb_dim,self.hiddenatt_dim)
        self.i2q2=nn.Linear(self.emb_dim,self.hiddenatt_dim)
        self.i2k2=nn.Linear(self.emb_dim,self.hiddenatt_dim)
        self.i2v2=nn.Linear(self.emb_dim,self.hiddenatt_dim)
        self.i2q3=nn.Linear(self.emb_dim,self.hiddenatt_dim)
        self.i2k3=nn.Linear(self.emb_dim,self.hiddenatt_dim)
        self.i2v3=nn.Linear(self.emb_dim,self.hiddenatt_dim)
        self.i2q4=nn.Linear(self.emb_dim,self.hiddenatt_dim)
        self.i2k4=nn.Linear(self.emb_dim,self.hiddenatt_dim)
        self.i2v4=nn.Linear(self.emb_dim,self.hiddenatt_dim)
        self.norm=nn.LayerNorm(self.hidden_dim,eps=1e-05,elementwise_affine=False)
        self.soft=nn.Softmax(dim=1)
    def forward(self,input_ids,is_first): #embedded [batch,len,emb_dim]
        if is_first==1:
            embedded=self.emb(input_ids)
            embedded=self.Position(embedded)
        else:
            embedded=input_ids
        q1=self.i2q1(embedded)
        k1=self.i2k1(embedded)
        v1=self.i2v1(embedded)
        k1=k1.permute(0,2,1)
        dot1=torch.matmul(q1,k1)
        dot1=dot1/self.k
        soft_layer1=self.soft(dot1)
        z1=torch.matmul(soft_layer1,v1)
        q2=self.i2q2(embedded)
        k2=self.i2k2(embedded)
        v2=self.i2v2(embedded)
        k2=k2.permute(0,2,1)
        dot2=torch.matmul(q2,k2)
        dot2=dot2/self.k
        soft_layer2=self.soft(dot2)
        z2=torch.matmul(soft_layer2,v2)
        q3=self.i2q3(embedded)
        k3=self.i2k3(embedded)
        v3=self.i2v3(embedded)
        k3=k3.permute(0,2,1)
        dot3=torch.matmul(q3,k3)
        dot3=dot3/self.k
        soft_layer3=self.soft(dot3)
        z3=torch.matmul(soft_layer3,v3)
        q4=self.i2q4(embedded)
        k4=self.i2k4(embedded)
        v4=self.i2v4(embedded)
        k4=k4.permute(0,2,1)
        dot4=torch.matmul(q4,k4)
        dot4=dot4/self.k
        soft_layer4=self.soft(dot4)
        z4=torch.matmul(soft_layer4,v4)
        z=torch.cat((z1,z2,z3,z4),dim=2)
        z=z+embedded
        z=self.norm(z)
        zz=self.ffn(z)
        zz=self.ffnlinear(zz)  # z [batch,len,hidden_dim]
        zz=self.norm(z+zz)
        return zz

class Attention(nn.Module):
    def __init__(self, config,Encoder1,Encoder2,Encoder3,tag2id):
        super(Attention, self).__init__()
        self.config = config
        self.device = config.device
        self.emb_dim = config.emb_dim
        self.tag2id=tag2id
        self.target_size = len(self.tag2id)
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.tag2id = config.tag2id
        self.en1=Encoder1
        self.en2=Encoder2
        self.en3=Encoder3
        self.h2t=nn.Linear(self.hidden_dim,self.target_size)
        self.loss= CrossEntropyLoss()
    def forward(self,inputs,labels, lengths, mask):

        #print("hid1",hid1.size())
        """这里可以把每个hidi向量加上一个跟位置有关的向量再进行后续处理"""
        hid1=self.en1(inputs,1)  #hidall [batch,len,hid_dim]
        hid2=self.en2(hid1,0)
        hid3=self.en3(hid2,0)
        #print(hidall.size(),"hidall")
        target=self.h2t(hid3)
        #print(target.size(),"target")
        pred_tag = torch.argmax(target, dim=-1)
        #print("pred_tag",pred_tag.size())
        target = target.view(-1, self.target_size)[mask.view(-1) == 1.0]
        loss = self.loss(target, labels.view(-1))
        return loss,pred_tag
