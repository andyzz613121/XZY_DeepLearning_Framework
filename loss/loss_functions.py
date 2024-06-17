import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter


class Classification_Loss():
    def loss_with_hed(input, edge, label, threshold=0.5):#std*sum
        n, c, h, w = input.size()
        # print(input.size())
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)
        # print(edge.shape)
        edge = edge.view(1, -1).float()
        # print(edge.shape)
        label = label.view(1, -1).long()
        # print(label.shape)
        weights_EDGE = edge.cpu().detach().numpy()
        weights_EDGE = torch.from_numpy(weights_EDGE).cuda()
        weights_EDGE = weights_EDGE/torch.max(weights_EDGE)

        weights_class = np.zeros((label.shape[0], label.shape[1])).astype(np.float32)
        weights_class = torch.from_numpy(weights_class).cuda()
        red_index = (label==2)
        pos_index = (label == 1)
        neg_index = (label == 0)
        red_num = red_index.sum().float()
        pos_num = pos_index.sum().float()
        neg_num = neg_index.sum().float()
        sum_num = pos_num + neg_num #这边直接这样pos_num dtype = int64,修改SegNet_fuseHED

        pos_rate = 1-(pos_num/sum_num)
        neg_rate = 1-(neg_num/sum_num)
        red_rate = 1-(red_num/sum_num)
    
        weights_class[pos_index] = pos_rate
        weights_class[neg_index] = neg_rate
        weights_class[red_index] = red_rate
        
        weights_class = weights_class/torch.max(weights_class)

        weight = torch.add(weights_class, weights_EDGE)
        m = nn.LogSoftmax(dim=1)
        labeln_1 = torch.unsqueeze(label[0], 1) #onehot need label shape (n, 1)
        one_hot = torch.zeros(label.shape[1], 3).cuda().scatter_(1, labeln_1, 1)
        one_hot = one_hot.transpose(0, 1)
        log_cross = m(log_cross)
        # print(one_hot.shape, log_cross[0].shape)
        loss_ce = torch.mul(one_hot, log_cross[0])
        
        loss_ce = torch.mul(loss_ce, weight[0])
        loss_ce = -1 * torch.sum(loss_ce)/label.shape[1]
        return loss_ce

    def loss_with_hed0(input, label, threshold=0.5):#median loss cause over bigger loss
        n, c, h, w = input.size()
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)
        # print(edge.shape)
        # edge = edge.view(1, 6, -1).float()
        # print(edge.shape)
        label = label.view(1, -1).long()
        # print(label.shape)
        weights = np.zeros((label.shape[0], label.shape[1])).astype(np.float32)

        #calculate the median class&edge class num
        classes_num = []

        for classes in range(6):
            class_index = (label==classes)
            class_num = class_index.sum().cpu().numpy()
            classes_num.append(class_num)

        classes_num = np.array(classes_num)
        sum_num = classes_num.sum()
        
        for classes in range(6):
            class_index = (label==classes).cpu()
            class_num = class_index.sum().cpu().numpy()
            if class_num != 0:
                weights[class_index] = 1 - class_num/sum_num #give class weight by median/class_num    

            # edge_class_index = (edge[:,classes,:]>threshold).cpu()
            #存在问题，一个点两个类都是大于0.5多次乘怎么办
            # weights[edge_class_index] *= 2

        weights_dict = Counter(weights[0])
        Cross_Entropy = 0
        for key in weights_dict:
            CE_pos_index = (weights == key)
            CE_pos_index = CE_pos_index.astype(bool)
            Cal_Pixel = log_cross[:,:,CE_pos_index[0]]
            Cal_target = label[:,CE_pos_index[0]]
            
            cross_Entropy0 = nn.CrossEntropyLoss()(Cal_Pixel, Cal_target)
            cross_Entropy0 *= key
            Cross_Entropy += cross_Entropy0
        print(Cross_Entropy)
        return Cross_Entropy

    def loss_with_hed1(input, edge, label, threshold=0.5):#std*sum
        n, c, h, w = input.size()
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)
        print(edge.shape)
        edge = edge.view(1, 6, -1).float()
        print(edge.shape)
        label = label.view(1, -1).long()
        print(label.shape)
        weights = np.zeros((label.shape[0], label.shape[1])).astype(np.float32)
        print(weights.shape)
        #calculate the median class&edge class num
        classes_num = []
        for classes in range(6):
            class_index = (label==classes)
            class_num = class_index.sum().cpu().numpy()
            classes_num.append(class_num)

        classes_num = np.array(classes_num)
        sum_num = classes_num.sum()
        
        min_class_num = 100000 #class 6 rate = min class num rate
        for classes in range(6):
            class_index = (label==classes).cpu()
            class_num = class_index.sum().cpu().numpy()
            if class_num < min_class_num:
                min_class_num = class_num
            if class_num != 0:
                weights[class_index] = 1 - class_num/sum_num #give class weight by median/class_num
            if class_num == 5:
                weights[class_index] = 1 - min_class_num/sum_num#class 6 rate = min class num rate

        # print(weights)    
        weights0 = weights
        # print(weights0, np.max(weights0), np.min(weights0))
        sum_prob = edge[0].cpu().detach().numpy().sum(0)
        edge_np = edge[0].cpu().detach().numpy()
        IE = np.zeros(edge.shape[2]).astype(np.float)
        for channel in range(6):
            zero_index = (edge_np[channel]==0)
            None_zero_index = (edge_np[channel]!=0)
            IE[zero_index] += 0
            # print(None_zero_index.sum())
            IE[None_zero_index] += -1*edge_np[channel][None_zero_index]*np.log(edge_np[channel][None_zero_index])
            # IE += -1*edge_np[channel]*np.log(edge_np[channel])
            # print(IE, np.max(IE), np.min(IE))
        # print(sum_prob.shape)
        over_threshold_index = (sum_prob>threshold)
        # print(sum_prob[over_threshold_index])
        # print(edge[0,0:6,over_threshold_index].cpu().detach().numpy().shape)
        # print(np.std(edge[0,0:6,over_threshold_index].cpu().detach().numpy(), 0).shape)
        # weights0[0][over_threshold_index] += sum_prob[over_threshold_index] * np.std(edge[0,0:6,over_threshold_index].cpu().detach().numpy(), 0)
        weights0[0][over_threshold_index] += IE[over_threshold_index]
        # print(weights0, np.max(weights0), np.min(weights0))
        weights0 = torch.from_numpy(weights0).cuda()
        
        # print(weights, np.max(weights), np.min(weights), np.var(weights))
        # a = 0
        # for pixel in range(edge.shape[2]):
        #     prob = edge[0,0:6,pixel].cpu().detach().numpy()
        #     sum_prob = prob.sum()
        #     if sum_prob > threshold:
        #         weights[0][pixel] += np.std(prob)*sum_prob
        #         # print(sum_prob, np.std(prob))
        #         a+=1
        # print(weights, np.max(weights), np.min(weights), np.var(weights))
        # print(a)
        m = nn.LogSoftmax(dim=1)
        loss_function = nn.NLLLoss()
        labeln_1 = torch.unsqueeze(label[0], 1) #onehot need label shape (n, 1)
        one_hot = torch.zeros(label.shape[1], 6).cuda().scatter_(1, labeln_1, 1)
        one_hot = one_hot.transpose(0, 1)
        log_cross = m(log_cross)
        loss_ce = torch.mul(one_hot, log_cross[0])
        loss_ce = torch.mul(loss_ce, weights0[0])
        loss_ce = -1 * torch.sum(loss_ce)/label.shape[1]
    

        # for pixel in range(label.shape[1]):
        #     label_class = label[0, pixel]
        #     loss_ce += log_cross[0, label_class, pixel]
        # loss_ce = -1*loss_ce/label.shape[1]
        
        # loss_ce = loss_function(log_cross, label)
        # print(loss_ce, loss_ce.shape)
        return loss_ce

    def loss_with_CB(input, edge, label, threshold=0.5):#std*sum
        n, c, h, w = input.size()

        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)
        edge = edge.view(1, c, -1).float()
        label = label.view(1, -1).long()
        weights = np.zeros((label.shape[0], label.shape[1])).astype(np.float32)
        
        #calculate the median class&edge class num
        classes_num = []
        for classes in range(c):
            class_index = (label==classes)
            class_num = class_index.sum().cpu().numpy()
            classes_num.append(class_num)

        classes_num = np.array(classes_num)
        sum_num = classes_num.sum()
        
        for classes in range(c):
            class_index = (label==classes).cpu()
            class_num = class_index.sum().cpu().numpy()
            rate = 1 - class_num/sum_num
            if class_num != 0:
                weights[class_index] = rate # give class weight by median/class_num
 
        weights0 = weights

        edge_np = edge[0].cpu().detach().numpy()
        sum_prob = edge_np.sum(0)
        IE = np.zeros(edge.shape[2]).astype(np.float)
        for channel in range(1):
            zero_index = (edge_np[channel]==0)
            None_zero_index = (edge_np[channel]!=0)
            IE[zero_index] += 0
            IE[None_zero_index] += -1*edge_np[channel][None_zero_index]*np.log(edge_np[channel][None_zero_index])

        over_threshold_index = (sum_prob>threshold)
        weights0[0][over_threshold_index] += IE[over_threshold_index]
        weights0 = torch.from_numpy(weights0).cuda()

        m = nn.LogSoftmax(dim=1)
        loss_function = nn.NLLLoss()
        labeln_1 = torch.unsqueeze(label[0], 1) #onehot need label shape (n, 1)
        one_hot = torch.zeros(label.shape[1], c).cuda().scatter_(1, labeln_1, 1)
        one_hot = one_hot.transpose(0, 1)
        log_cross = m(log_cross)
        loss_ce = torch.mul(one_hot, log_cross[0])
        loss_ce = torch.mul(loss_ce, weights0[0])
        loss_ce = -1 * torch.sum(loss_ce)/label.shape[1]
    
        return loss_ce
    
    def loss_with_hed_pos(input, edge, label, threshold=0.5):#std*sum
        n, c, h, w = input.size()
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)
        edge = edge.view(1, 6, -1).float()
        label = label.view(1, -1).long()
        weights = np.zeros((label.shape[0], label.shape[1])).astype(np.float32)
        
        #calculate the median class&edge class num
        classes_num = []
        for classes in range(6):
            class_index = (label==classes)
            class_num = class_index.sum().cpu().numpy()
            classes_num.append(class_num)

        classes_num = np.array(classes_num)
        sum_num = classes_num.sum()
        
        min_class_num = 100000 #class 6 rate = min class num rate
        for classes in range(6):
            class_index = (label==classes).cpu()
            class_num = class_index.sum().cpu().numpy()
            if class_num < min_class_num:
                min_class_num = class_num
            if class_num != 0:
                weights[class_index] = 1 - class_num/sum_num #give class weight by median/class_num

        # print(weights)    
        weights0 = weights
        # print(weights0, np.max(weights0), np.min(weights0))
        sum_prob = edge[0].cpu().detach().numpy().sum(0)
        edge_np = edge[0].cpu().detach().numpy()
        IE = np.zeros(edge.shape[2]).astype(np.float)
        for channel in range(6):
            zero_index = (edge_np[channel]==0)
            None_zero_index = (edge_np[channel]!=0)
            IE[zero_index] += 0
            IE[None_zero_index] += -1*edge_np[channel][None_zero_index]*np.log(edge_np[channel][None_zero_index])

        over_threshold_index = (sum_prob>threshold)
        
        weights0[0][over_threshold_index] += IE[over_threshold_index]
        # print(weights0, np.max(weights0), np.min(weights0))
        weights0 = torch.from_numpy(weights0).cuda()

        m = nn.LogSoftmax(dim=1)
        loss_function = nn.NLLLoss()
        labeln_1 = torch.unsqueeze(label[0], 1) #onehot need label shape (n, 1)
        one_hot = torch.zeros(label.shape[1], 6).cuda().scatter_(1, labeln_1, 1)
        one_hot = one_hot.transpose(0, 1)
        log_cross = m(log_cross)
        loss_ce = torch.mul(one_hot, log_cross[0])
        loss_ce = torch.mul(loss_ce, weights0[0])
        loss_ce = -1 * torch.sum(loss_ce)/label.shape[1]
    

        return loss_ce

    def loss_with_CB_RB(input, edge, label, threshold=0.5):#std*sum
        n, c, h, w = input.size()
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)
        # print(edge.shape)
        edge = edge.view(1, 3, -1).float()
        # print(edge.shape)
        label = label.view(1, -1).long()
        # print(label.shape)
        weights = np.zeros((label.shape[0], label.shape[1])).astype(np.float32)
        # print(weights.shape)
        #calculate the median class&edge class num
        classes_num = []
        for classes in range(3):
            class_index = (label==classes)
            class_num = class_index.sum().cpu().numpy()
            classes_num.append(class_num)

        classes_num = np.array(classes_num)
        sum_num = classes_num.sum()
        
        min_class_num = 100000 #class 3 rate = min class num rate
        for classes in range(3):
            class_index = (label==classes).cpu()
            class_num = class_index.sum().cpu().numpy()
            if class_num < min_class_num:
                min_class_num = class_num
            if class_num != 0:
                weights[class_index] = 1 - class_num/sum_num #give class weight by median/class_num

        # print(weights)    
        weights0 = weights
        # print(weights0, np.max(weights0), np.min(weights0))
        sum_prob = edge[0].cpu().detach().numpy().sum(0)
        edge_np = edge[0].cpu().detach().numpy()
        IE = np.zeros(edge.shape[2]).astype(np.float)
        for channel in range(3):
            zero_index = (edge_np[channel]==0)
            None_zero_index = (edge_np[channel]!=0)
            IE[zero_index] += 0
            IE[None_zero_index] += -1*edge_np[channel][None_zero_index]*np.log(edge_np[channel][None_zero_index])
    
        over_threshold_index = (sum_prob>threshold)
        
        weights0[0][over_threshold_index] += IE[over_threshold_index]
        weights0 = torch.from_numpy(weights0).cuda()
        

        m = nn.LogSoftmax(dim=1)
        loss_function = nn.NLLLoss()
        labeln_1 = torch.unsqueeze(label[0], 1) #onehot need label shape (n, 1)
        one_hot = torch.zeros(label.shape[1], 3).cuda().scatter_(1, labeln_1, 1)
        one_hot = one_hot.transpose(0, 1)
        log_cross = m(log_cross)
        loss_ce = torch.mul(one_hot, log_cross[0])
        loss_ce = torch.mul(loss_ce, weights0[0])
        loss_ce = -1 * torch.sum(loss_ce)/label.shape[1]
    

        return loss_ce

    def loss_with_hed_RB(input, edge, label, threshold=0.5):#std*sum
        n, c, h, w = input.size()
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)
        # print(edge.shape)
        edge = edge.view(1, 1, -1).float()
        # print(edge.shape)
        label = label.view(1, -1).long()
        # print(label.shape)
        weights = np.zeros((label.shape[0], label.shape[1])).astype(np.float32)
        # print(weights.shape)
        #calculate the median class&edge class num
        classes_num = []
        for classes in range(3):
            class_index = (label==classes)
            class_num = class_index.sum().cpu().numpy()
            classes_num.append(class_num)

        classes_num = np.array(classes_num)
        sum_num = classes_num.sum()
        
        min_class_num = 100000 #class 3 rate = min class num rate
        for classes in range(3):
            class_index = (label==classes).cpu()
            class_num = class_index.sum().cpu().numpy()
            if class_num < min_class_num:
                min_class_num = class_num
            if class_num != 0:
                weights[class_index] = 1 - class_num/sum_num #give class weight by median/class_num

        # print(weights)    
        weights0 = weights
        # print(weights0, np.max(weights0), np.min(weights0))
        sum_prob = edge[0].cpu().detach().numpy().sum(0)
        edge_np = edge[0].cpu().detach().numpy()
        IE = np.zeros(edge.shape[2]).astype(np.float)
        for channel in range(1):
            zero_index = (edge_np[channel]==0)
            None_zero_index = (edge_np[channel]!=0)
            IE[zero_index] += 0
            IE[None_zero_index] += -1*edge_np[channel][None_zero_index]*np.log(edge_np[channel][None_zero_index])
    
        over_threshold_index = (sum_prob>threshold)
        
        weights0[0][over_threshold_index] += IE[over_threshold_index]
        weights0 = torch.from_numpy(weights0).cuda()
        

        m = nn.LogSoftmax(dim=1)
        loss_function = nn.NLLLoss()
        labeln_1 = torch.unsqueeze(label[0], 1) #onehot need label shape (n, 1)
        one_hot = torch.zeros(label.shape[1], 3).cuda().scatter_(1, labeln_1, 1)
        one_hot = one_hot.transpose(0, 1)
        log_cross = m(log_cross)
        loss_ce = torch.mul(one_hot, log_cross[0])
        loss_ce = torch.mul(loss_ce, weights0[0])
        loss_ce = -1 * torch.sum(loss_ce)/label.shape[1]
    

        return loss_ce

    def loss_with_class_old(input, label, threshold=0.5):#std*sum
        n, c, h, w = input.size()
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)
        # print(edge.shape)
        label = label.view(1, -1).long()
        # print(label.shape)
        weights = np.zeros((label.shape[0], label.shape[1])).astype(np.float32)
        # print(weights.shape)
        #calculate the median class&edge class num
        classes_num = []
        for classes in range(c):
            class_index = (label==classes)
            class_num = class_index.sum().cpu().numpy()
            classes_num.append(class_num)

        classes_num = np.array(classes_num)
        sum_num = classes_num.sum()
        
        min_class_num = 100000 #class 3 rate = min class num rate
        for classes in range(c):
            class_index = (label==classes).cpu()
            class_num = class_index.sum().cpu().numpy()
            if class_num < min_class_num:
                min_class_num = class_num
            if class_num != 0:
                weights[class_index] = 1 - class_num/sum_num #give class weight by median/class_num

        # print(weights)    
        weights0 = weights
        weights0 = torch.from_numpy(weights0).cuda()
        
        m = nn.LogSoftmax(dim=1)
        loss_function = nn.NLLLoss()
        labeln_1 = torch.unsqueeze(label[0], 1) #onehot need label shape (n, 1)
        one_hot = torch.zeros(label.shape[1], c).cuda().scatter_(1, labeln_1, 1)
        one_hot = one_hot.transpose(0, 1)
        log_cross = m(log_cross)
        loss_ce = torch.mul(one_hot, log_cross[0])
        loss_ce = torch.mul(loss_ce, weights0[0])
        loss_ce = -1 * torch.sum(loss_ce)/label.shape[1]

        return loss_ce

    def loss_with_class_nonorm(input, label, ignore_label=-1):
        n, c, h, w = input.size()
        
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)
        label = label.view(1, -1).long()
        weights = np.zeros((label.shape[0], label.shape[1])).astype(np.float32)

        #calculate the median class&edge class num
        classes_num = []
        for classes in range(c):
            class_index = (label==classes)
            class_num = class_index.sum().cpu().numpy()
            classes_num.append(class_num)

        classes_num = np.array(classes_num)
        sum_num = classes_num.sum()

        max_rate = 0
        for classes in range(c):
            class_index = (label==classes).cpu()
            class_num = class_index.sum().cpu().numpy()
            rate = 1 - class_num/sum_num
            if rate > max_rate:
                max_rate = rate
            if class_num != 0:
                weights[class_index] = rate # give class weight by median/class_num

        if ignore_label != -1:
            class_index = (label==ignore_label).cpu()
            weights[class_index] = 2*max_rate

        weights0 = weights
        weights0 = torch.from_numpy(weights0).cuda()

        m = nn.LogSoftmax(dim=1)
        loss_function = nn.NLLLoss()
        labeln_1 = torch.unsqueeze(label[0], 1) #onehot need label shape (n, 1)
        one_hot = torch.zeros(label.shape[1], c).cuda().scatter_(1, labeln_1, 1)
        one_hot = one_hot.transpose(0, 1)
        log_cross = m(log_cross)
        loss_ce = torch.mul(one_hot, log_cross[0])
        loss_ce = torch.mul(loss_ce, weights0[0])
        loss_ce = -1 * torch.sum(loss_ce)/label.shape[1]

        return loss_ce
    
    def loss_with_class_norm(input, label, ignore_label=-1, zhiding_weight=None):
        '''
            ignore_label 处的权重为最大权重（即等于数量最少的类的权重）
        '''
        n, c, h, w = input.size()
        label_torch = label
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)
        label = label.view(1, -1).long()
        if zhiding_weight != None:
            zhiding_weight = torch.tensor(zhiding_weight).cuda()
            loss_function = nn.CrossEntropyLoss(weight=zhiding_weight)
            loss_ce = loss_function(input, label_torch.long())
        else:
            weights = torch.zeros((label.shape[0], label.shape[1])).cuda()
            #calculate the median class&edge class num
            classes_num = []
            for classes in range(c):
                class_index = (label==classes)
                class_num = class_index.sum()
                classes_num.append(class_num)

            classes_num = torch.tensor(classes_num).cuda()
            sum_num = classes_num.sum()
            class_rate = classes_num/sum_num
            class_rate = 1-torch.nn.functional.softmax(class_rate)
            for classes in range(c):
                class_index = (label==classes)
                weights[class_index] = class_rate[classes] # give class weight by median/class_num

            if ignore_label != -1:
                class_index = (label==ignore_label)
                weights[class_index] = max(class_rate)

            # 自己计算的带权loss，和Pytorch官方一样的：
            # m = nn.LogSoftmax(dim=1)
            # labeln_1 = torch.unsqueeze(label[0], 1) #onehot need label shape (n, 1)
            # one_hot = torch.zeros(label.shape[1], c).cuda().scatter_(1, labeln_1, 1)
            # one_hot = one_hot.transpose(0, 1)
            # log_cross = m(log_cross)
            # loss_ce = torch.mul(one_hot, log_cross[0])
            # loss_ce = torch.mul(loss_ce, weights[0])
            # Pytorch 计算加权loss除以的是一张图里所有weight加起来的值，而不是像元数，即第二行的
            # # loss_ce = -1 * torch.sum(loss_ce)/label.shape[1]
            # loss_ce = -1 * torch.sum(loss_ce)/weights.view(1, -1).sum(1)
            loss_function = nn.CrossEntropyLoss(weight=class_rate)
            loss_ce = loss_function(input, label_torch.long())
        return loss_ce

    def Siamese_loss(dsm_out, img_out, margin=2.0):
        n, c, h, w = dsm_out.size()
        dsm_out = F.softmax(dsm_out, dim=1)
        img_out = F.softmax(img_out, dim=1)

        dsm_out = dsm_out.transpose(0, 1).contiguous().view(1, c, -1)
        img_out = img_out.transpose(0, 1).contiguous().view(1, c, -1)
        _, classes, pixel_num = dsm_out.size()

        #相同的赋值为1，不同的为0
        similarity = torch.zeros(1, pixel_num).float().cuda()
        dsm_argmax = torch.argmax(dsm_out, axis=1)
        img_argmax = torch.argmax(img_out, axis=1)
        same_index = (dsm_argmax==img_argmax)
        # print(dsm_argmax, img_argmax)
        # print(same_index.sum())
        similarity[same_index] = 1
        # print(img_out, dsm_out)
        dis = F.pairwise_distance(img_out, dsm_out, keepdim=True)
        # print(torch.max(dis))
        loss_first = 0.5 * similarity * dis * dis
        # print(torch.max(loss_first), torch.min(loss_first))
        loss_last = 0.5 * (1-similarity) * torch.clamp(margin-dis, min=0.0) * torch.clamp(margin-dis, min=0.0)
        # print(torch.max(loss_last), torch.min(loss_last))
        loss = loss_first + loss_last
        
        loss = torch.mean(loss)

        return loss

class Contrastive_Loss():
    def info_nce_loss_base(features, n_views=2, temperature=0.07):
        '''
        Usage: 对比损失，最小化正样本距离，最大化负样本距离。
        Input: 
               Features: 正负样本特征( 一组的shape为[B, C] )，共n_views组，即[B, C, n_views]
               n_views:  正样本生成的对数
               temperature: 温度参数
        Output: 
               logits: 两列，第一列为正样本对的相似度, 第二列为正样本对所有负样本的相似度
               labels: 全都是0, 表示输入特征为第一列正样本相似度以及第二列负样本相似度的情况下，输出结果为0。例如第一列为1，后面为2...2，则训练后所有的输入特征都要满足这种情况，
               才能使labels为0，从而达到控制正样本与负样本之间距离的目的。
        example:
               In: Batch: B, n_views: 2
               logits: 2*B
               labels: 2*B - 2 (减去自身，以及自身生成的负样本对)

        '''
        b_nviews, c = features.size()
        batch = b_nviews / n_views
        torch.set_printoptions(profile='full')

        labels = torch.cat([torch.arange(batch) for i in range(n_views)], dim=0)   # [b*n_views]
            # print('labels1', labels.shape)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().cuda()    # [b*n_views, b*n_views]
            # print('labels2', labels.shape, labels)

            # print('features', features.shape, features.dtype)
        features = F.normalize(features, dim=1)                                 # [b*n_views, C]
            # print('features', features.shape)
        similarity_matrix = torch.matmul(features, features.T)                  # [b*n_views, b*n_views]
            # print('similarity_matrix', similarity_matrix.shape)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()              # [b*n_views, b*n_views]
        labels = labels[~mask].view(labels.shape[0], -1)                        # [b*n_views, b*n_views - 1]
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)      # [b*n_views, b*n_views - 1]
        # assert similarity_matrix.shape == labels.shape
            # print('mask', mask.shape)
            # print('labels3', labels.shape)
            # print('similarity_matrix3', similarity_matrix.shape)
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [b*n_views, 1]
            # print('positives', positives.shape)
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # [b*n_views, b*n_views - 2]
            # print('negatives', negatives.shape)
        logits = torch.cat([positives, negatives], dim=1)                          # [b*n_views, b*n_views - 1]
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()             # [b*n_views]
            # print('logits', logits.shape)
            # print('labels4', labels.shape)
        logits = logits / temperature
        return logits, labels

    def info_nce_loss_withclass(features, label, other_class_num = 10, temperature=0.07):
        '''
        Usage: 对比损失，最小化正样本距离，最大化负样本距离。区别于info_nce_loss_base，这里的每一个类相对于其他类为负样本，相对于自己类为正样本，如果有多个正样本，取均值
        Input: 
               Features: 正负样本特征( 一组的shape为[B, C] )，共n_views组，即[B, C, n_views]
               labels: 类别标签[B]
               other_class_num:  负样本类别的个数
               temperature: 温度参数
        Output: 
               logits: 两列，第一列为正样本对的相似度, 第二列为正样本对所有负样本的相似度
               labels: 全都是0, 表示输入特征为第一列正样本相似度以及第二列负样本相似度的情况下，输出结果为0。例如第一列为1，后面为2...2，则训练后所有的输入特征都要满足这种情况，
               才能使labels为0，从而达到控制正样本与负样本之间距离的目的。
        example:
               In: Batch: B, n_views: 2
               logits: 2*B
               labels: 2*B - 2 (减去自身，以及自身生成的负样本对)

        '''
        torch.set_printoptions(profile='full')

        # 找到一个batch中，哪些类是相同的。labels为[batch, batch]的矩阵，其中相同类的为1，其它为0
        # print('labels1', label)
        labels = (label.unsqueeze(0) == label.unsqueeze(1)).float().cuda()    # [b*n_views, b*n_views]
            # print('labels2', labels)shou

        # 计算相似度矩阵
        features = F.normalize(features, dim=1)                                 # [b*n_views, C]
        similarity_matrix = torch.matmul(features, features.T)                  # [b*n_views, b*n_views]
            # print('similarity_matrix', similarity_matrix)

        # 生成一个单位阵
        # 对于正样本来说，应该选择labels里面除去单位阵的等于1的元素
        # 对于负样本来说，应该选择1-labels里面等于1的元素
        mask = torch.eye(labels.shape[0]).cuda()                                # [b*n_views, b*n_views]
        poslabels_mask = labels - mask                                          # [b*n_views, b*n_views - 1]
        neglabels_mask = 1 - labels
    
        # 选择正样本
        positives = similarity_matrix*poslabels_mask
        positives_num = poslabels_mask.bool().sum(1)
        # print(positives_num.sum(), positives_num)
        if positives_num.sum() == 0:
            return None, None

        # print('positives_num', positives_num.shape, positives_num)
        # print('positives', positives.shape, positives)
        
        # 对于一个batch中只出现过一次的类别，删除
        pair_class_mask = (positives_num > 0)
        positives = positives[pair_class_mask]
        positives_num = positives_num[pair_class_mask]
            # print('positives_num', positives_num.shape, positives_num)
        positives_means = positives.sum(1) / positives_num
            # print('positives_means', positives_means.shape, positives_means)

        # 选择负样本，对于一个batch中只出现过一次的类别，删除
        negatives = similarity_matrix*neglabels_mask
            # print('negatives1', negatives.shape, negatives)
        negatives = negatives[pair_class_mask]
            # print('negatives2', negatives.shape, negatives)
        negatives, _ = torch.sort(negatives, 1, descending=True)
            # print('negatives3', negatives.shape, negatives)
        # 对负样本进行排序，选择最接近的other_class_num值
        negatives = negatives[:, 0:other_class_num]
            # print('negatives4', negatives.shape, negatives)
        logits = torch.cat([positives_means.view(-1, 1), negatives], dim=1)        # [b*n_views, b*n_views - 1]
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()             # [b*n_views]
            # print('logits', logits.shape, logits)
            # print('labels4', labels.shape)
        logits = logits / temperature
        return logits, labels

    # def info_nce_loss(features, labels, n_views=1, temperature=0.07):
    #     '''
    #     Input:  Features(B, C) 输入的向量
    #             n_views： 图像对的个数
    #     '''
    #     torch.set_printoptions(profile='full') 
    #     # labels = torch.cat([torch.arange(features.shape[0]) for i in range(n_views)], dim=0)        # [b*n_views]
    #     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().cuda()                        # [b*n_views, b*n_views]
    #     features = F.normalize(features, dim=1)                                                   
    #     similarity_matrix = torch.matmul(features, features.T)
    #     # assert similarity_matrix.shape == (
    #     #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    #     # assert similarity_matrix.shape == labels.shape

    #     # discard the main diagonal from both: labels and similarities matrix
    #     mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    #     labels = labels[~mask].view(labels.shape[0], -1)
    #     similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    #     # assert similarity_matrix.shape == labels.shape

    #     # select and combine multiple positives
    #     positives = similarity_matrix[labels.bool()].view(-1, 1)
    #     # select only the negatives the negatives
    #     negatives = similarity_matrix[~labels.bool()].view(-1, 1)

    #     logits = torch.cat([positives, negatives], dim=0)
    #     logits = logits / temperature

    #     labels_pos = torch.ones([positives.shape[0], 1], dtype=torch.long).cuda()
    #     labels_neg = torch.zeros([negatives.shape[0], 1], dtype=torch.long).cuda()
    #     labels = torch.cat([labels_pos, labels_neg], dim=0).view(-1)
    #     print(F.cross_entropy(logits, labels))
    #     # labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
    #     return logits, labels
        
class Segment_Loss():
    def __init__(self, criterion=0):
        '''
        图像分割的损失函数
        Input: 
            criterion: 衡量标准（0 距离；1 信息熵）

        '''
        self.criterion = criterion
    
    def IE_loss(self, img):

        return 0
    
    def DIS_loss(self, img):
        img_mask = torch.argmax(img, 1)
        img_onehot = torch.nn.functional.one_hot(img_mask, self.Seg_Num).permute(0, 3, 1, 2)
        return 0


    def seg_loss(self, img):
        if self.criterion == 0:
            return self.DIS_loss(img)
        if self.criterion == 1:
            return self.IE_loss(img)


class HED_Loss():
    def HED_LOSS(input, target):
        n, c, h, w = input.size()
        
        # assert(max(target) == 1)
        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.unsqueeze(1).transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()
        pos_index = (target_t > 0)
        neg_index = (target_t == 0)
        target_trans[pos_index] = 1
        target_trans[neg_index] = 0
        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        if pos_num==0:
            print('111')
            return False
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num*1.0 / sum_num
        weight[neg_index] = pos_num*1.0 / sum_num
        weight = weight/np.max(weight)
        # print(pos_num,neg_num)
        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        # print(log_p.shape, target_t.shape)
        loss = F.binary_cross_entropy(log_p, target_t, weight, size_average=True)
        return loss

    def HED_LOSS_WITH_DISTANCE(input, target, class_label):  
        n, c, h, w = input.size()
        #用于回归的input，在类别维度上把所有值相加(还可以进行1*1卷积)
        input_mse = input.sum(1).view(n,1,h,w)

        log_mse = input_mse.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).float()
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)

        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).float()
        target_cross = class_label.view(1, -1)
        
        weights = target_t.clone()
        weights = weights.cpu().numpy().astype(np.float32)
        
        np.set_printoptions(threshold=np.inf)
        pos_index = (target_t > 0)
        neg_index = (target_t == 0)

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num

        if pos_num == 0:
            return False

        weights[pos_index] *= (neg_num*1.0 / sum_num)
        weights[neg_index] = pos_num*1.0 / sum_num

        weights = weights/np.max(weights) #可修改
        weights_dict = Counter(weights[0])
        
        Cross_Entropy = 0
        for key in weights_dict:
            CE_pos_index = (weights == key)
            CE_pos_index = CE_pos_index.astype(bool)
            Cal_Pixel = log_cross[:,:,CE_pos_index[0]]
            Cal_target = target_cross[:,CE_pos_index[0]]
            
            cross_Entropy0 = nn.CrossEntropyLoss()(Cal_Pixel, Cal_target)
            cross_Entropy0 = cross_Entropy0 * key
            Cross_Entropy += cross_Entropy0

        weights = torch.from_numpy(weights).cuda()
        mse = (log_mse - target_t) * (log_mse - target_t) * weights
        #边界crossEntropy权重与非边界一致
        #cross_Entropy0 = nn.CrossEntropyLoss()(log_cross, target_cross)
        # print('CE0 = %f'%cross_Entropy0)
        # cross_Entropy1 = 0
        # for i in range(log_cross.shape[2]):
        #     weight_CE = F.cross_entropy(log_cross[:,:,i], target_cross[:,i]) * weights[:,i]
        #     cross_Entropy1 += weight_CE
        # print(log_cross.shape[2])
        # print(cross_Entropy1)

        
        #rate = int(Cross_Entropy/(torch.sum(mse)/sum_num))
        rate = 10
        mse = (torch.sum(mse)/sum_num) * rate
        #print(rate, torch.sum(mse)/sum_num,mse, Cross_Entropy)
        return mse + Cross_Entropy

    def HED_LOSS_WITH_DISTANCE_AND_CLASS_RATE_Potsdam(input, target, class_label):  
        n, c, h, w = input.size()
        #用于回归的input，在类别维度上把所有值相加(还可以进行1*1卷积)
        input_mse = input.sum(1).view(n,1,h,w)
        log_mse = input_mse.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).float()
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)

        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).float()
        target_class = class_label.view(1, -1).long()
        
        weights = target_t.clone()
        weights = weights.cpu().numpy().astype(np.float32)
        
        np.set_printoptions(threshold=np.inf)

        pos_index = (target_t > 0)
        neg_index = (target_t == 0)
        ground_index = (target_class == 0)
        building_index = (target_class == 1)
        LowVegetation_index = (target_class == 2)
        Tree_index = (target_class == 3)
        car_index = (target_class == 4)
        background_index = (target_class == 5)

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ground_index = ground_index.data.cpu().numpy().astype(bool)
        building_index = building_index.data.cpu().numpy().astype(bool)
        LowVegetation_index = LowVegetation_index.data.cpu().numpy().astype(bool)
        Tree_index = Tree_index.data.cpu().numpy().astype(bool)
        car_index = car_index.data.cpu().numpy().astype(bool)
        background_index = background_index.data.cpu().numpy().astype(bool)

        ground_index = pos_index * ground_index
        building_index = pos_index * building_index
        LowVegetation_index = pos_index * LowVegetation_index
        Tree_index = pos_index * Tree_index
        car_index = pos_index * car_index
        background_index = pos_index *background_index

        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        ground_num = ground_index.sum()
        building_num = building_index.sum()
        LowVegetation_num = LowVegetation_index.sum()
        Tree_num = Tree_index.sum()
        car_num = car_index.sum()
        background_num = background_index.sum()
        sum_class_num = ground_num + building_num + LowVegetation_num + Tree_num + car_num + background_num
        #print(pos_num, sum_class_num, ground_num, building_num, LowVegetation_num, Tree_num, car_num, background_num)
        sum_num = pos_num + neg_num
        
        ground_rate = 1 - (ground_num/sum_class_num)
        building_rate = 1 - (building_num/sum_class_num)
        LowVegetation_rate = 1 - (LowVegetation_num/sum_class_num)
        Tree_rate = 1 - (Tree_num/sum_class_num)
        car_rate = 1 - (car_num/sum_class_num)
        background_rate = 1 - (background_num/sum_class_num)
        #print(ground_rate,building_rate,LowVegetation_rate,Tree_rate,car_rate,except_bg_rate)
        #print(ground_rate+building_rate+LowVegetation_rate+Tree_rate+car_rate)
        if pos_num == 0:
            return False, False

        weights[pos_index] *= (neg_num*1.0 / sum_num)
        weights[neg_index] = pos_num*1.0 / sum_num

        weights[ground_index] *= ground_rate
        weights[building_index] *= building_rate
        weights[LowVegetation_index] *= LowVegetation_rate
        weights[Tree_index] *= Tree_rate
        weights[car_index] *= car_rate
        weights[background_index] *= background_rate
        
        weights = weights/np.max(weights) #可修改
        weights_dict = Counter(weights[0])

        Cross_Entropy = 0
        for key in weights_dict:
            CE_pos_index = (weights == key)
            CE_pos_index = CE_pos_index.astype(bool)
            Cal_Pixel = log_cross[:,:,CE_pos_index[0]]
            Cal_target = target_class[:,CE_pos_index[0]]
            
            cross_Entropy0 = nn.CrossEntropyLoss()(Cal_Pixel, Cal_target)
            cross_Entropy0 *= key
            Cross_Entropy += cross_Entropy0

        weights = torch.from_numpy(weights).cuda()

        mse = (log_mse - target_t) * (log_mse - target_t) *  weights
        rate_mse = int(Cross_Entropy/(torch.sum(mse)/sum_num))
        rate_ce = int(((torch.sum(mse)/sum_num))/Cross_Entropy)
        # rate = 10
        if rate_ce < rate_mse:
            mse = (torch.sum(mse)/sum_num) * rate_mse
        elif rate_ce > rate_mse:
            Cross_Entropy *= rate_ce
            print('1')
        return mse + Cross_Entropy
        #return (torch.sum(mse)/sum_num)

    def HED_LOSS_WITH_DISTANCE_AND_CLASS_RATE_RB(input, target, class_label):  
        n, c, h, w = input.size()
        #用于回归的input，在类别维度上把所有值相加(还可以进行1*1卷积)
        input_mse = input.sum(1).view(n,1,h,w)
        log_mse = input_mse.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).float()
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)

        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).float()
        target_class = class_label.view(1, -1).long()
        
        weights = target_t.clone()
        weights = weights.cpu().numpy().astype(np.float32)
        
        np.set_printoptions(threshold=np.inf)

        pos_index = (target_t > 0)
        neg_index = (target_t == 0)
        ground_index = (target_class == 0)
        building_index = (target_class == 1)
        LowVegetation_index = (target_class == 2)

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ground_index = ground_index.data.cpu().numpy().astype(bool)
        building_index = building_index.data.cpu().numpy().astype(bool)
        LowVegetation_index = LowVegetation_index.data.cpu().numpy().astype(bool)

        ground_index = pos_index * ground_index
        building_index = pos_index * building_index
        LowVegetation_index = pos_index * LowVegetation_index

        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        ground_num = ground_index.sum()
        building_num = building_index.sum()
        LowVegetation_num = LowVegetation_index.sum()
        sum_class_num = ground_num + building_num + LowVegetation_num
        #print(pos_num, sum_class_num, ground_num, building_num, LowVegetation_num, Tree_num, car_num, background_num)
        sum_num = pos_num + neg_num
        
        ground_rate = 1 - (ground_num/sum_class_num)
        building_rate = 1 - (building_num/sum_class_num)
        LowVegetation_rate = 1 - (LowVegetation_num/sum_class_num)
        #print(ground_rate,building_rate,LowVegetation_rate,Tree_rate,car_rate,except_bg_rate)
        #print(ground_rate+building_rate+LowVegetation_rate+Tree_rate+car_rate)
        if pos_num == 0:
            return False, False

        # print(np.max(weights), np.min(weights))
        weights[pos_index] *= (neg_num*1.0 / sum_num)
        weights[neg_index] = pos_num*1.0 / sum_num
        # print(np.max(weights), np.min(weights))
        weights[ground_index] *= ground_rate
        weights[building_index] *= building_rate
        weights[LowVegetation_index] *= LowVegetation_rate
        # print(np.max(weights[ground_index]), np.min(weights[ground_index]))
        # print(np.max(weights[building_index]), np.min(weights[building_index]))
        # print(np.max(weights[LowVegetation_index]), np.min(weights[LowVegetation_index]))

        weights = weights/np.max(weights) #可修改
        weights_dict = Counter(weights[0])

        Cross_Entropy = 0
        for key in weights_dict:
            CE_pos_index = (weights == key)
            CE_pos_index = CE_pos_index.astype(bool)
            Cal_Pixel = log_cross[:,:,CE_pos_index[0]]
            Cal_target = target_class[:,CE_pos_index[0]]
            
            cross_Entropy0 = nn.CrossEntropyLoss()(Cal_Pixel, Cal_target)
            cross_Entropy0 *= key
            Cross_Entropy += cross_Entropy0

        weights = torch.from_numpy(weights).cuda()

        mse = (log_mse - target_t) * (log_mse - target_t) *  weights
        rate_mse = int(Cross_Entropy/(torch.sum(mse)/sum_num))
        rate_ce = int(((torch.sum(mse)/sum_num))/Cross_Entropy)
        # rate = 10
        if rate_ce < rate_mse:
            mse = (torch.sum(mse)/sum_num) * rate_mse
        elif rate_ce > rate_mse:
            Cross_Entropy *= rate_ce
            print('1')
        return mse + Cross_Entropy

    def HED_LOSS_WITH_DISTANCE_AND_CLASS_RATE(input, target, class_label):  
        n, c, h, w = input.size()
        #用于回归的input，在类别维度上把所有值相加(还可以进行1*1卷积)
        input_mse = input.sum(1).view(n,1,h,w)
        log_mse = input_mse.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).float()
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)

        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).float()
        target_class = class_label.view(1, -1).long()
        
        weights = target_t.clone()
        weights = weights.cpu().numpy().astype(np.float32)
        
        np.set_printoptions(threshold=np.inf)

        pos_index = (target_t > 0)
        neg_index = (target_t == 0)
        ground_index = (target_class == 0)
        building_index = (target_class == 1)
        LowVegetation_index = (target_class == 2)
        Tree_index = (target_class == 3)
        car_index = (target_class == 4)
        background_index = (target_class == 5)

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ground_index = ground_index.data.cpu().numpy().astype(bool)
        building_index = building_index.data.cpu().numpy().astype(bool)
        LowVegetation_index = LowVegetation_index.data.cpu().numpy().astype(bool)
        Tree_index = Tree_index.data.cpu().numpy().astype(bool)
        car_index = car_index.data.cpu().numpy().astype(bool)
        background_index = background_index.data.cpu().numpy().astype(bool)

        ground_index = pos_index * ground_index
        building_index = pos_index * building_index
        LowVegetation_index = pos_index * LowVegetation_index
        Tree_index = pos_index * Tree_index
        car_index = pos_index * car_index
        background_index = pos_index *background_index

        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        ground_num = ground_index.sum()
        building_num = building_index.sum()
        LowVegetation_num = LowVegetation_index.sum()
        Tree_num = Tree_index.sum()
        car_num = car_index.sum()
        background_num = background_index.sum()
        sum_class_num = ground_num + building_num + LowVegetation_num + Tree_num + car_num + background_num
        #print(pos_num, sum_class_num, ground_num, building_num, LowVegetation_num, Tree_num, car_num, background_num)
        sum_num = pos_num + neg_num
        if sum_class_num == 0:
            print('11111111111111111111111')
        except_bg_rate = 1 - background_num/sum_class_num
        if except_bg_rate == 0:
            print('22222222222222222222222',background_num, sum_class_num)
        ground_rate = 1 - (ground_num/sum_class_num)
        building_rate = 1 - (building_num/sum_class_num)
        LowVegetation_rate = 1 - (LowVegetation_num/sum_class_num)
        Tree_rate = 1 - (Tree_num/sum_class_num)
        car_rate = 1 - (car_num/sum_class_num)
        background_rate = np.max(np.array([ground_rate, building_rate, LowVegetation_rate, Tree_rate, car_rate]))
        # print(ground_rate,building_rate,LowVegetation_rate,Tree_rate,car_rate,background_rate)
        #print(ground_rate+building_rate+LowVegetation_rate+Tree_rate+car_rate)
        if pos_num == 0 or sum_class_num == 0 or except_bg_rate == 0:
            print(class_label)
            print(torch.max(target_class),torch.min(target_class))
            return False


        weights[pos_index] *= (neg_num*1.0 / sum_num)
        weights[neg_index] = pos_num*1.0 / sum_num

        weights[ground_index] *= ground_rate
        weights[building_index] *= building_rate
        weights[LowVegetation_index] *= LowVegetation_rate
        weights[Tree_index] *= Tree_rate
        weights[car_index] *= car_rate
        weights[background_index] *= background_rate
        weights = weights/np.max(weights) #可修改
        weights_dict = Counter(weights[0])

        Cross_Entropy = 0
        for key in weights_dict:
            CE_pos_index = (weights == key)
            CE_pos_index = CE_pos_index.astype(bool)
            Cal_Pixel = log_cross[:,:,CE_pos_index[0]]
            Cal_target = target_class[:,CE_pos_index[0]]
            
            cross_Entropy0 = nn.CrossEntropyLoss()(Cal_Pixel, Cal_target)
            cross_Entropy0 *= key
            Cross_Entropy += cross_Entropy0

        weights = torch.from_numpy(weights).cuda()

        mse = (log_mse - target_t) * (log_mse - target_t) *  weights
        rate_mse = int(Cross_Entropy/(torch.sum(mse)/sum_num))
        rate_ce = int(((torch.sum(mse)/sum_num))/Cross_Entropy)
        # rate = 10
        if rate_ce < rate_mse:
            mse = (torch.sum(mse)/sum_num) * rate_mse
        elif rate_ce > rate_mse:
            Cross_Entropy *= rate_ce
            print('1')
        return mse + Cross_Entropy
        #return (torch.sum(mse)/sum_num)

    def HED_LOSS_WITH_DISTANCE_AND_CLASS_RATE_MUTIL_TASK(input, target, class_label):  
        n, c, h, w = input.size()
        #用于回归的input，在类别维度上把所有值相加(还可以进行1*1卷积)
        input_mse = input.sum(1).view(n,1,h,w)

        log_mse = input_mse.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).float()
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)

        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).float()
        target_class = class_label.view(1, -1).long()
        
        weights = target_t.clone()
        weights = weights.cpu().numpy().astype(np.float32)
        
        np.set_printoptions(threshold=np.inf)

        pos_index = (target_t > 0)
        neg_index = (target_t == 0)
        ground_index = (target_class == 0)
        building_index = (target_class == 1)
        LowVegetation_index = (target_class == 2)
        Tree_index = (target_class == 3)
        car_index = (target_class == 4)
        background_index = (target_class == 5)

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ground_index = ground_index.data.cpu().numpy().astype(bool)
        building_index = building_index.data.cpu().numpy().astype(bool)
        LowVegetation_index = LowVegetation_index.data.cpu().numpy().astype(bool)
        Tree_index = Tree_index.data.cpu().numpy().astype(bool)
        car_index = car_index.data.cpu().numpy().astype(bool)
        background_index = background_index.data.cpu().numpy().astype(bool)

        ground_index = pos_index * ground_index
        building_index = pos_index * building_index
        LowVegetation_index = pos_index * LowVegetation_index
        Tree_index = pos_index * Tree_index
        car_index = pos_index * car_index
        background_index = pos_index *background_index

        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        ground_num = ground_index.sum()
        building_num = building_index.sum()
        LowVegetation_num = LowVegetation_index.sum()
        Tree_num = Tree_index.sum()
        car_num = car_index.sum()
        background_num = background_index.sum()
        sum_class_num = ground_num + building_num + LowVegetation_num + Tree_num + car_num + background_num
        #print(pos_num, sum_class_num, ground_num, building_num, LowVegetation_num, Tree_num, car_num, background_num)
        sum_num = pos_num + neg_num
        
        ground_rate = 1 - (ground_num/sum_class_num)
        building_rate = 1 - (building_num/sum_class_num)
        LowVegetation_rate = 1 - (LowVegetation_num/sum_class_num)
        Tree_rate = 1 - (Tree_num/sum_class_num)
        car_rate = 1 - (car_num/sum_class_num)
        background_rate = np.max(np.array([ground_rate, building_rate, LowVegetation_rate, Tree_rate, car_rate]))
        #print(ground_rate,building_rate,LowVegetation_rate,Tree_rate,car_rate,except_bg_rate)
        #print(ground_rate+building_rate+LowVegetation_rate+Tree_rate+car_rate)
        if pos_num == 0:
            return False, False

        weights[pos_index] *= (neg_num*1.0 / sum_num)
        weights[neg_index] = pos_num*1.0 / sum_num

        weights[ground_index] *= ground_rate
        weights[building_index] *= building_rate
        weights[LowVegetation_index] *= LowVegetation_rate
        weights[Tree_index] *= Tree_rate
        weights[car_index] *= car_rate
        weights[background_index] *= background_rate

        weights = weights/np.max(weights) #可修改
        weights_dict = Counter(weights[0])

        Cross_Entropy = torch.tensor(0.0).cuda()
        for key in weights_dict:
            CE_pos_index = (weights == key)
            CE_pos_index = CE_pos_index.astype(bool)
            Cal_Pixel = log_cross[:,:,CE_pos_index[0]]
            Cal_target = target_class[:,CE_pos_index[0]]
            
            cross_Entropy0 = nn.CrossEntropyLoss()(Cal_Pixel, Cal_target)
            cross_Entropy0 *= key
            Cross_Entropy += cross_Entropy0
        
        weights = torch.from_numpy(weights).cuda()
        mse = (log_mse - target_t) * (log_mse - target_t) * weights
        mse = (torch.sum(mse)/sum_num)
        
        return mse, Cross_Entropy

    def HED_LOSS_WITH_NO_DISTANCE_AND_CLASS_RATE(input, target, class_label):  
        n, c, h, w = input.size()
        #用于回归的input，在类别维度上把所有值相加(还可以进行1*1卷积)
        input_cross_edge = input.sum(1).view(n,1,h,w)
        log_cross_edge = input_cross_edge.transpose(0, 1).contiguous().view(1, 1, -1).float()
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1).float()
        print(log_cross.shape)
        print(log_cross_edge.shape)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).long()
        target_class = class_label.view(1, -1).long()
        
        weights = target_t.clone()
        weights = weights.cpu().numpy().astype(np.float32)
        target_t[target_t>0]=1
        np.set_printoptions(threshold=np.inf)
        
        pos_index = (target_t > 0)
        neg_index = (target_t == 0)
        ground_index = (target_class == 0)
        building_index = (target_class == 1)
        LowVegetation_index = (target_class == 2)
        Tree_index = (target_class == 3)
        car_index = (target_class == 4)
        background_index = (target_class == 5)

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ground_index = ground_index.data.cpu().numpy().astype(bool)
        building_index = building_index.data.cpu().numpy().astype(bool)
        LowVegetation_index = LowVegetation_index.data.cpu().numpy().astype(bool)
        Tree_index = Tree_index.data.cpu().numpy().astype(bool)
        car_index = car_index.data.cpu().numpy().astype(bool)
        background_index = background_index.data.cpu().numpy().astype(bool)

        ground_index = pos_index * ground_index
        building_index = pos_index * building_index
        LowVegetation_index = pos_index * LowVegetation_index
        Tree_index = pos_index * Tree_index
        car_index = pos_index * car_index
        background_index = pos_index *background_index

        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        ground_num = ground_index.sum()
        building_num = building_index.sum()
        LowVegetation_num = LowVegetation_index.sum()
        Tree_num = Tree_index.sum()
        car_num = car_index.sum()
        background_num = background_index.sum()
        sum_class_num = ground_num + building_num + LowVegetation_num + Tree_num + car_num + background_num
        
        sum_num = pos_num + neg_num
        if sum_class_num == 0:
            print('11111111111111111111111')
        except_bg_rate = 1 - background_num/sum_class_num
        if except_bg_rate == 0:
            print('22222222222222222222222',background_num, sum_class_num)
        ground_rate = 1 - (ground_num/sum_class_num)
        building_rate = 1 - (building_num/sum_class_num)
        LowVegetation_rate = 1 - (LowVegetation_num/sum_class_num)
        Tree_rate = 1 - (Tree_num/sum_class_num)
        car_rate = 1 - (car_num/sum_class_num)
        background_rate = np.max(np.array([ground_rate, building_rate, LowVegetation_rate, Tree_rate, car_rate]))
        
        if pos_num == 0 or sum_class_num == 0 or except_bg_rate == 0:
            print(class_label)
            print(torch.max(target_class),torch.min(target_class))
            return False


        weights[pos_index] *= (neg_num*1.0 / sum_num)
        weights[neg_index] = pos_num*1.0 / sum_num

        weights[ground_index] *= ground_rate
        weights[building_index] *= building_rate
        weights[LowVegetation_index] *= LowVegetation_rate
        weights[Tree_index] *= Tree_rate
        weights[car_index] *= car_rate
        weights[background_index] *= background_rate
        weights = weights/np.max(weights) #可修改
        weights_dict = Counter(weights[0])

        Cross_Entropy = 0
        Cross_Entropy_edge = 0
        for key in weights_dict:
            CE_pos_index = (weights == key)
            CE_pos_index = CE_pos_index.astype(bool)
            Cal_Pixel = log_cross[:,:,CE_pos_index[0]]
            Cal_target = target_class[:,CE_pos_index[0]]
            cross_Entropy0 = nn.CrossEntropyLoss()(Cal_Pixel, Cal_target)
            cross_Entropy0 *= key
            Cross_Entropy += cross_Entropy0
            # print(log_mse.shape)
            # print(target_t.shape)
            Cal_Pixel_edge = log_cross_edge[:,:,CE_pos_index[0]]
            Cal_target_edge = target_t[:,CE_pos_index[0]]
            cross_Entropy1 = nn.CrossEntropyLoss()(Cal_Pixel_edge, Cal_target_edge)
            cross_Entropy1 *= key
            Cross_Entropy_edge += cross_Entropy1

        rate_edge_ce = int(Cross_Entropy/Cross_Entropy_edge)
        rate_ce = int(Cross_Entropy_edge/Cross_Entropy)
        print('rate_ce, rate_eg_ce', rate_ce, rate_edge_ce)

        if Cross_Entropy < Cross_Entropy_edge:
            Cross_Entropy_edge = Cross_Entropy_edge * rate_edge_ce
        else:
            Cross_Entropy *= rate_ce

        return Cross_Entropy_edge + Cross_Entropy

    def HED_LOSS_WITH_DISTANCE_AND_CLASS_RATE_Suichang(input, target, class_label):  
        n, c, h, w = input.size()
        #用于回归的input，在类别维度上把所有值相加(还可以进行1*1卷积)
        input_mse = input.sum(1).view(n,1,h,w)
        log_mse = input_mse.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).float()
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)

        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).float()
        target_class = class_label.view(1, -1).long()
        
        weights = target_t.clone()
        weights = weights.cpu().numpy().astype(np.float32)
        
        np.set_printoptions(threshold=np.inf)

        pos_index = (target_t > 0)
        neg_index = (target_t == 0)
        index0 = (target_class == 0)
        index1 = (target_class == 1)
        index2 = (target_class == 2)
        index3 = (target_class == 3)
        index4 = (target_class == 4)
        index5 = (target_class == 5)
        index6 = (target_class == 6)
        index7 = (target_class == 7)
        index8 = (target_class == 8)
        index9 = (target_class == 9)

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        index0 = index0.data.cpu().numpy().astype(bool)
        index1 = index1.data.cpu().numpy().astype(bool)
        index2 = index2.data.cpu().numpy().astype(bool)
        index3 = index3.data.cpu().numpy().astype(bool)
        index4 = index4.data.cpu().numpy().astype(bool)
        index5 = index5.data.cpu().numpy().astype(bool)
        index6 = index6.data.cpu().numpy().astype(bool)
        index7 = index7.data.cpu().numpy().astype(bool)
        index8 = index8.data.cpu().numpy().astype(bool)
        index9 = index9.data.cpu().numpy().astype(bool)

        index0 = pos_index * index0
        index1 = pos_index * index1
        index2 = pos_index * index2
        index3 = pos_index * index3
        index4 = pos_index * index4
        index5 = pos_index * index5
        index6 = pos_index * index6
        index7 = pos_index * index7
        index8 = pos_index * index8
        index9 = pos_index * index9

        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        num0 = index0.sum()
        num1 = index1.sum()
        num2 = index2.sum()
        num3 = index3.sum()
        num4 = index4.sum()
        num5 = index5.sum()
        num6 = index6.sum()
        num7 = index7.sum()
        num8 = index8.sum()
        num9 = index9.sum()
        
        sum_class_num = num0+num1+num2+num3+num4+num5+num6+num7+num8+num9
        #print(pos_num, sum_class_num, ground_num, building_num, LowVegetation_num, Tree_num, car_num, background_num)
        sum_num = pos_num + neg_num
        
        rate0 = 1 - (num0/sum_class_num)
        rate1 = 1 - (num1/sum_class_num)
        rate2 = 1 - (num2/sum_class_num)
        rate3 = 1 - (num3/sum_class_num)
        rate4 = 1 - (num4/sum_class_num)
        rate5 = 1 - (num5/sum_class_num)
        rate6 = 1 - (num6/sum_class_num)
        rate7 = 1 - (num7/sum_class_num)
        rate8 = 1 - (num8/sum_class_num)
        rate9 = 1 - (num9/sum_class_num)
        if pos_num == 0:
            return False, False

        weights[pos_index] *= (neg_num*1.0 / sum_num)
        weights[neg_index] = pos_num*1.0 / sum_num

        weights[index0] *= rate0
        weights[index1] *= rate1
        weights[index2] *= rate2
        weights[index3] *= rate3
        weights[index4] *= rate4
        weights[index5] *= rate5
        weights[index6] *= rate6
        weights[index7] *= rate7
        weights[index8] *= rate8
        weights[index9] *= rate9
        
        weights = weights/np.max(weights) #可修改
        weights_dict = Counter(weights[0])

        Cross_Entropy = 0
        for key in weights_dict:
            CE_pos_index = (weights == key)
            CE_pos_index = CE_pos_index.astype(bool)
            Cal_Pixel = log_cross[:,:,CE_pos_index[0]]
            Cal_target = target_class[:,CE_pos_index[0]]
            
            cross_Entropy0 = nn.CrossEntropyLoss()(Cal_Pixel, Cal_target)
            cross_Entropy0 *= key
            Cross_Entropy += cross_Entropy0

        weights = torch.from_numpy(weights).cuda()

        mse = (log_mse - target_t) * (log_mse - target_t) *  weights
        rate_mse = int(Cross_Entropy/(torch.sum(mse)/sum_num))
        rate_ce = int(((torch.sum(mse)/sum_num))/Cross_Entropy)
        # rate = 10
        if rate_ce < rate_mse:
            mse = (torch.sum(mse)/sum_num) * rate_mse
        elif rate_ce > rate_mse:
            Cross_Entropy *= rate_ce
            print('1')
        return mse + Cross_Entropy
        #return (torch.sum(mse)/sum_num)

    def HED_LOSS_WITH_DISTANCE_AND_CLASS_RATE_MUTIL_TASK_GENERAL(input, target, class_label, class_num):  
        n, c, h, w = input.size()
        #用于回归的input，在类别维度上把所有值相加(还可以进行1*1卷积)
        input_mse = input.sum(1).view(n,1,h,w)

        log_mse = input_mse.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).float()
        log_cross = input.transpose(0, 1).contiguous().view(1, c, -1)

        target_t = target.unsqueeze(1).transpose(1, 2).transpose(2, 3).contiguous().view(1, -1).float()
        target_class = class_label.view(1, -1).long()
        
        weights = target_t.clone()
        weights = weights.cpu().numpy().astype(np.float32)
        
        np.set_printoptions(threshold=np.inf)

        # 边界的正负像素点, 及个数
        pos_index = (target_t > 0)
        neg_index = (target_t == 0)
        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)

        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_edge_num = pos_num + neg_num

        if pos_num == 0:
            return False
        
        # 分类的正负像素点，并与边界像素点合并，及其个数，比例
        idx_list = []
        num_list = []
        rate_list = []
        for i in range(class_num):
            idx = (target_class == i)
            idx = idx.data.cpu().numpy().astype(bool)
            # idx = idx * pos_index  # 与边界像素点合并
            idx_list.append(idx)

            num = idx.sum()
            num_list.append(num)
        sum_class_num = sum(num_list)
        
        for i in range(class_num):
            rate_list.append(1-num_list[i]/sum_class_num)

        # 计算权重（边界和类别相乘）
        weights[pos_index] = neg_num*1.0 / sum_edge_num
        weights[neg_index] = pos_num*1.0 / sum_edge_num
        for i in range(class_num):
            weights[idx_list[i]] *= rate_list[i]

        weights = weights/np.max(weights) #可修改

        # import sys
        # base_path = '..\\XZY_DeepLearning_Framework\\'
        # sys.path.append(base_path)
        # from data_processing.Raster import gdal_write_tif
        # weights = weights.reshape((-1, 256, 256))
        # gdal_write_tif('D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\Edge\\111.tif', weights[0], 256, 256, 1, datatype=2)
        # gdal_write_tif('D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\Edge\\222.tif', target[0].cpu().detach().numpy(), 256, 256, 1, datatype=2)

        weights_dict = Counter(weights[0])
        Cross_Entropy = torch.tensor(0.0).cuda()
        for key in weights_dict:
            CE_pos_index = (weights == key)
            CE_pos_index = CE_pos_index.astype(bool)
            Cal_Pixel = log_cross[:,:,CE_pos_index[0]]
            Cal_target = target_class[:,CE_pos_index[0]]
            
            cross_Entropy0 = nn.CrossEntropyLoss()(Cal_Pixel, Cal_target)
            cross_Entropy0 *= key
            Cross_Entropy += cross_Entropy0
        
        weights = torch.from_numpy(weights).cuda()
        mse = (log_mse - target_t) * (log_mse - target_t) * weights
        mse = (torch.sum(mse)/sum_edge_num)
        
        rate_mse = int(Cross_Entropy/mse)
        rate_ce = int(mse/Cross_Entropy)
        rate = 10
        # print(rate_ce, rate_mse)
        if rate_ce < rate_mse:
            mse *= rate_mse
        elif rate_ce > rate_mse:
            Cross_Entropy *= rate_ce
        # print(mse, Cross_Entropy)
        return mse + 4*Cross_Entropy

    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def focal_loss_zhihu(input, target):
    '''
    :param input: 使用知乎上面大神给出的方案  https://zhuanlan.zhihu.com/p/28527749
    :param target:
    :return:
    '''
    n, c, h, w = input.size()

    target = target.long()
    inputs = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.contiguous().view(-1)

    N = inputs.size(0)
    C = inputs.size(1)

    weights = torch.tensor((1, 10), dtype=torch.float32).cuda()
    weights=weights[target.view(-1)]#这行代码非常重要

    gamma = 2

    P = F.softmax(inputs, dim=1)#shape [num_samples,num_classes]
    class_mask = inputs.data.new(N, C).fill_(0)
    class_mask = Variable(class_mask)
    ids = target.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)#shape [num_samples,num_classes]  one-hot encoding
    probs = (P * class_mask).sum(1).view(-1, 1)#shape [num_samples,]
    log_p = probs.log()

    print('in calculating batch_loss',weights.shape,probs.shape,log_p.shape)

    batch_loss = -weights * (torch.pow((1 - probs), gamma)) * log_p
    # batch_loss = -(torch.pow((1 - probs), gamma)) * log_p

    print(batch_loss.shape)

    loss = batch_loss.mean()
    return loss

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
 
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
 
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class focal_loss(nn.Module):
    def __init__(self, model, alpha, gamma=2, num_classes = 2, size_average=True):
        super(focal_loss,self).__init__()
        if model == 1:
            self.model = 21.05
        if model == 2:
            self.model = 4.52
        if model == 3:
            self.model = 8.31
        if model == 4:
            self.model = 1

        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重,
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类,
            
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes],

        self.gamma = gamma
        print(alpha)
        print(self.model)
        print(self.gamma)
    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1,
        # print(preds.shape, labels.shape)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax,
        preds_softmax = torch.exp(preds_logsoft)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll ),
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.cuda()
        self.alpha = self.alpha.gather(0,labels.view(-1).cuda()).cuda()

        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ,
        loss = torch.mul(self.alpha, loss.t())

        if self.size_average:
            loss = loss.mean() * self.model
        else:
            loss = loss.sum() * self.model
        return loss

if __name__ == '__main__':
    Loss = Loss_With_Edge()
    Loss.loss_with_CB_RB
    