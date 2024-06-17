import torch
import torch.nn as nn
class lstm_with_classification(nn.Module):
    '''
        Input: 
                input_size: 输入的向量特征长度
                hidden_size: lstm内隐藏层的特征长度
                num_layers: lstm cell层的个数
                num_classes: 分类的类别数
        Forward Input: 在lstm中batch_first 为 True的情况下
                x: (批大小batch_size, 序列长度seq_len, 特征长度feature_len=input_size) 
                h0, c0: (num_layers*num_directions(单向或双向), 批大小batch_size, hidden_size)
                out: (batch_size, seq_length, hidden_size)
    '''
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(lstm_with_classification, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) #, batch_first=True
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x):
        # # Set initial hidden and cell states
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        # # Forward propagate LSTM
        # out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # # Decode the hidden state of the last time step
        # out = self.fc(out[:, -1, :])# 此处的-1说明我们只取RNN最后输出的那个hn
        # return out

        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(self.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[-1, :, :])# 此处的-1说明我们只取RNN最后输出的那个hn
        return out

if __name__ == '__main__':
    # sequence_length = 289
    # feat_len = 28
    # hidden_size = 128
    # num_layers = 2
    # num_classes = 10
    # batch_size = 101

    # aa = torch.zeros([batch_size, sequence_length, feat_len])
    # l = lstm_with_classification(feat_len, hidden_size, num_layers, num_classes)
    # out = l(aa)
    # print(out.shape)

    # 准备数据
    input_size = 10   # 输入特征数
    hidden_size = 20  # 隐藏层特征数
    num_layers = 2    # LSTM层数
    output_size = 2   # 输出类别数
    batch_size = 3    # 批大小
    sequence_length = 5  # 序列长度
    
    # 随机生成一些数据
    x = torch.randn(sequence_length, batch_size, input_size).cuda()
    y = torch.randint(output_size, (batch_size,)).cuda()
    print(x.shape, y.shape)
    # 定义优化器和损失函数
    model = lstm_with_classification(input_size, hidden_size, num_layers, output_size).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 开始训练
    num_epochs = 100
    for epoch in range(num_epochs):
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(outputs.shape)
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    
    
    # 预测新数据
    with torch.no_grad():
        for i in range(100):
            test_x = torch.randn(sequence_length, batch_size, input_size).cuda()
            outputs = model(test_x)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
