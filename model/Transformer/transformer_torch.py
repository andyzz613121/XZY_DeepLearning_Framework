import math
from typing import Tuple
 
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
 
 
# 词向量的位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        '''
        :param d_model: 词向量的嵌入维度
        :param dropout: dropout概率
        :param max_len:一句话的最大长度
        '''
        super(PositionalEncoding, self).__init__()
        # dropout层
        self.dropout = nn.Dropout(p=dropout)
        # 生成从0到max_len-1的位置索引，维度为[max_len]
        # 添加一个第一维度
        # position=[max_len,1]
        position = torch.arange(max_len).unsqueeze(1)
        # 按照论文《Attention is all you need》中的位置编码计算公式
        # 注意这里为什么torch.arange(0,d_model,2)每隔2个生成一个
        # 因为后面要分奇偶位计算位置编码向量
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # 位置编码向量
        pe = torch.zeros(max_len, 1, d_model)
        # 分别为每句话的奇数位置，偶数位置赋予不同的位置编码
        # torch.sin(position*div_term)=[max_len,d_model/2]
        # 因为position=[max_len，1]
        # dir_term=[d_model/2]
        # 这里存在一个广播机制
        # 所以词向量的维数必须是偶数
        # 要不然会赋值失败
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # 保存位置编码向量到模型的参数记录中
        self.register_buffer('pe', pe)
 
    def forward(self, x: Tensor) -> Tensor:
        # 返回带有位置编码的词嵌入向量
        # x=[句子的长度，批大小，词嵌入维度]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
 
 
# 产生词向量的Mask矩阵
def generate_square_subsequent_mask(sz: int) -> Tensor:
    # 生成一个上三角Mask矩阵
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
 
 
# 定义Transformer结构
class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        '''
        :param ntoken:词典中词语的个数
        :param d_model:词嵌入向量维度
        :param nhead:多头注意力头数
        :param d_hid:Transformer编码层中隐藏层的神经单元数
        :param nlayers:编码器层数
        :param dropout:dropout概率
        '''
        super(TransformerModel, self).__init__()
        # 定义模型类型
        self.model_type = 'Transformer'
        # 进行位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # Transformer编码层
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        # 堆叠编码层形成编码器
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # 词嵌入
        self.encoder = nn.Embedding(ntoken, d_model)
        # 词嵌入维度信息
        self.d_model = d_model
        # 解码器
        self.decoder = nn.Linear(d_model, ntoken)
        # 初始化模型权重
        self.init_weights()
 
    def init_weights(self) -> None:
        initrange = 0.1
        # 正态分布初始化权重
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        # 偏置置为0
        self.decoder.bias.data.zero_()
 
    # 前向传递，建立计算图
    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        # src=[词语的个数，批大小]
        # src_mask=[词语的个数，词语的个数](Mask矩阵)
        # 进行词嵌入
        src = self.encoder(src) * math.sqrt(self.d_model)
        # 进行位置编码
        src = self.pos_encoder(src)
        # 进行Transformer编码
        output = self.transformer_encoder(src, src_mask)
        # 解码
        output = self.decoder(output)
        return output
 
 
# 类似于torchvision是计算机视觉的工具包
# torchtext是NLP的工具包
# wikitext2是Wikitext-103的子集用于测试语言模型的训练效果
from torchtext.datasets import WikiText2
# 分词工具，从序列中分出每一个词语
from torchtext.data.utils import get_tokenizer
# 建立词典工具，从数据集的迭代器中建立当前数据集的词典
from torchtext.vocab import build_vocab_from_iterator
 
# 获取WiKitext2的训练集
# 共有36718行
# root指定数据集存放的路径
train_iter = WikiText2(root='./data', split='train')
# 按照英文格式，以空格分割序列，其中有个tokenizer参数可以指定分割器
tokenizer = get_tokenizer('basic_english')
# 先将训练集中的词语分割出来，然后根据分割出来的这些词语建立一个词典，指定特殊词语用<unk>代替
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
# 为特殊标记<unk>添加索引
# 词典中的每一个词语都会关联一个索引
vocab.set_default_index(vocab['<unk>'])
 
 
def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    # 数据集处理
    # for item in rwa_test_iter迭代数据集的每一行文本
    # torkenizer(item)将这行文本转换为词语序列
    # vocab(tokenizer(item))返回词语序列中每个词语在词典中的索引
    # vocab(tokenizer(item))的结果是一个索引列表
    # torch.tensor()将索引列表转换为张量
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    # numel()函数统计一个张量中元素的总个数
    # filter()是一个过滤器
    # filter(lambda t:t.numel()>0,data)的意思是只获取data中元素个数大于0的元素
    # 因为data里面包含[]这样的元素，过滤掉
    # tuple()转换为元组才能进行torch.cat()拼接
    # torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))将所有的词语索引拼在一个列表里
    # 最后的维度是[N],N是词语的索引个数
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
 
 
# 获取训练集，验证集，测试集
train_iter, val_iter, test_iter = WikiText2()
# 处理数据集
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)
# 当前设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
 
def batchify(data: Tensor, bsz: int) -> Tensor:
    '''
    :param data: 词语序列
    :param bsz: 批大小
    :return:
    '''
    # 输入长度为N的词语序列
    # 转换为[N/batch_size,batch_size]的矩阵
    # 保证能够整除
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    # t()是矩阵转置
    # contiguous()是使分散在不同内存区域中的元素聚合在同一块连续的内存
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)
 
 
# 设置批大小
batch_size = 20
# 设置验证集批大小
eval_batch_size = 10
# 转换为[序列长度，批大小]的形式
# shape [seq_len, batch_size]
train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)
 
# 意义后面会解释
bptt = 35
 
 
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    # source=[full_seq_len,batch_size]
    # i是进行切分的位置
    # 将输入的数据切分成能够训练Transformer的[data,label]对
    # 其中data=[seq_len,batch_size]
    # label=[seq_len*batch_size]
    # 后面要切分，防止越界
    seq_len = min(bptt, len(source) - 1 - i)
    # 切分
    data = source[i:i + seq_len]
    # 这个reshape(-1)就是将target的维度转换为[seq_len*batch_size]
    # 为什么要i+1
    # 因为我们希望语言模型能够学习到词语的上下文表示
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target
 
 
# 定义参数
# 词典大小
ntokens = len(vocab)
# 词嵌入向量维度
emsize = 200
# 编码层的前馈神经网络的维度
d_hid = 200
# 编码器中编码层的层数
nlayers = 2
# 多头注意力头数
nhead = 2
# dropout概率
dropout = 0.2
# 实例化一个Transformer
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
import copy
import time
 
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 学习率
lr = 5.0  
# 随机梯度下降算法优化网络参数
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# 学习率衰减器
# 每隔1轮
# lr=0.95*lr
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
 
 
# 训练Transformer
def train(model: nn.Module) -> None:
    # 开启模型训练模式
    # model.train()开启模型的训练模式
    # model.eval()开启模型的验证模式
    # 验证模式下禁止反向传播，dropout层失效，训练模式相反
    model.train()
    # 损失值
    total_loss = 0.
    # 记录信息的间隔
    log_interval = 200
    # 开始训练的时间
    start_time = time.time()
    # 获取Mask矩阵
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    # 总共能进行训练的批次
    num_batches = len(train_data) // bptt
    # train_data.size(0)获取原先训练集的seq_len,
    # 为什么要-1
    # 因为后面我们需要从train_data中获取一对(data,label)用于训练
    # 而label是由data+1产生的
    # 所以最后到train_data.size(0)-1
    # 每隔bptt取一次训练对(data,label)
    # 到这里回顾一下数据集的变化
    # 最原始的数据集形态是一行一行的文本
    # ‘我 爱 你 中 国’（举例子，原数据集是英文的）
    # 然后我们对于这个数据集建立了一个词语到索引的一个词典,
    # 比如 我->346
    # 然后我们将数据集转换为了dataset=[3,532,4345,5,65434,...]这种形式，
    # 其中的每一个数字都是原先在数据集中词语在词典中的索引
    # 然后我们将dataset转换为了dataset=[N/batch_size,batch_size]这种形式
    # 因为我们要训练模型
    # 训练模型就得有样本和标记
    # 我们从dataset=[N/batch_size,batch_size]中每隔bptt抽取一对训练样本(data,label)
    # 其中data=[N/batch_size/bptt,batch_size]
    # label=[N/batch_size/bptt*batch_size]
    # 但是label我们是让data往后移动1位获取的
    # 这种处理方式能让Transformer学习到词语之间的上下文关系
    # 然后我们利用每一对(data,label)训练Transformer
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # 从切分点i获取一对训练对(data,label)
        data, targets = get_batch(train_data, i)
        # 获取当前训练对(data,label)中
        # data的seq_len
        batch_size = data.size(0)
        # 如果不等于我们制定的seq_len
        # 只可能发生在训练的最后一对(data,label)中
        if batch_size != bptt:
            # 需要改变Mask矩阵的大小
            # 否则不匹配
            src_mask = src_mask[:batch_size, :batch_size]

        # 获取Transforme对于当前的词语
        # 预测的紧接着它的下一个词语
        output = model(data, src_mask)
        # 我们的目的是想要Transformer学习到词语的上下文信息
        # 所以我们以这个目的作为为损失函数
        # 让Transformer预测这个词语真实的下一个词语的概率最大
        loss = criterion(output.view(-1, ntokens), targets)
        # 方向传播之前，梯度要清零
        # 如果不清零，梯度是累加的
        # 会导致结果不对
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 截断那些大于0.5的梯度值
        # 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # 更新模型参数
        optimizer.step()
        # 记录损失
        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            # 获取学习率
            # 因为我们设置了学习率衰减器
            # 学习率是动态变化的
            lr = scheduler.get_last_lr()[0]
            # 训练一对(data,label)所花费的时间
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            # 每对(data,label)的平均损失
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            # 打印信息
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            # 清零，重新记录
            total_loss = 0
            start_time = time.time()
 
 
# 验证Transformer
def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    # 开启验证模式
    model.eval()
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            batch_size = data.size(0)
            if batch_size != bptt:
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += batch_size * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)
 
 
# 最佳损失
best_val_loss = float('inf')
# 训练轮数
epochs = 3
# 最优模型
best_model = None
# 进行多epoch训练
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    # 训练Transformer
    train(model)
    # 验证，获取损失值
    val_loss = evaluate(model, val_data)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    # 打印信息
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)
    # 更新最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # deepcopy()会逐层的赋值model的参数
        # 深拷贝
        best_model = copy.deepcopy(model)
    # 学习率衰减
    # 注意，忘了从PyTorch 1.几之后了
    # 约定
    # 先optimizer优化网络参数
    # 然后再scheduler.step()衰减学习率
    # 否则会报错
    scheduler.step()
 
# 最后Transformer的self.encoder的参数保存了词嵌入矩阵
# 可以拿来进行下游的NLP任务
# print(model.encoder.parameter())
