import torch
import torch.nn as nn
import torch.nn.functional as F

def center_pixel(x):
    b, l, h, w = x.size()

    # 得到光谱图
    pth = int((h - 1)/2)
    ptw = int((w - 1)/2)
    pt = x[:,:,pth,ptw].view(b, -1)

    return pt

def center_img(x, window_size=1):
    b, l, h, w = x.size()

    # 得到光谱图
    pth = int((h - 1)/2)
    ptw = int((w - 1)/2)
    img_split = x[:,:,pth-window_size:pth+window_size+1,ptw-window_size:ptw+window_size+1]
    # print(img_split.shape, window_size)
    return img_split

# //////////////////////////////////////////////////////////////
# Resample Spectral
def MS_spectral(spec_vec, rsp_rate_list=[2]):
    '''
    Input:  spec_vec(spectral vector for one pixel)
            rsp_rate_list(The list of resample rates)

    Output: List
            resample_band(The list of resample bands) e.g.: [ [B, L], [B, L/2], [B, L/4]... ]  
    '''
    b, l = spec_vec.size()
    resample_band_list = []

    for rsp_rate in rsp_rate_list:
        resample_band = []
        for band in range(0, l, rsp_rate):
            resample_band.append(spec_vec[:, band].unsqueeze(1))
        resample_band = torch.cat([x for x in resample_band], 1)
        resample_band_list.append(resample_band)

    return resample_band_list

# Resample Spectral
def MS_spectral_entireimage(Hyper_Image, rsp_rate_list=[2]):
    '''
    Input:  Hyper_Image(Hyperspectral image) -> (B, C, H, W)
            rsp_rate_list(The list of resample rates)

    Output: List
            Image with resample_band(The list of image with resample bands) e.g.: [ [B, L, H, W], [B, L/2, H, W], [B, L/4, H, W]... ]  
    '''
    b, c, h, w = Hyper_Image.size()
    resample_band_list = []

    for rsp_rate in rsp_rate_list:
        resample_band = []
        for band in range(0, c, rsp_rate):
            resample_band.append(Hyper_Image[:, band, :, :].unsqueeze(1))
        resample_band = torch.cat([x for x in resample_band], 1)
        resample_band_list.append(resample_band)
    return resample_band_list

def MS_spectral_FeatureExtract(MS_spectral, act_list1, act_list2, act='2D', samesize=False):
        '''
        Usage:
                Extract Feature for Multiscale spectral
        Input: 
                MS_spectral [scale1, scale2, scale3] scale1 is original scale 
        Output: List
                MS_Feat(A list, Multiscale features)  [[B, 1, H, W], [B, 1, H/2, W/2]...]
        '''
        b, l = MS_spectral[0].size()

        # 计算多尺度特征
        MS_Feat = []
        for scale in range(len(MS_spectral)):
            spec_vec = MS_spectral[scale].unsqueeze(1)
            if act == 'MLP':
                Feat = compute_auto_mulply(spec_vec, act_list1[scale], act_list2[scale], 0)
            elif act == '1D':
                Feat = compute_spatt_conv1D(spec_vec, act_list1[scale], act_list2[scale], samesize=samesize)
            elif act == '2D':
                Feat = compute_spatt_conv2D(spec_vec, act_list1[scale], act_list2[scale], samesize=samesize)
            elif act == '1D2D':
                Feat = compute_spatt_conv1D2D(spec_vec, act_list1[scale], act_list2[scale], samesize=samesize)
            else:
                return False
            if samesize==True:
                Feat = F.interpolate(Feat, size=(l, l), mode='bilinear')

            MS_Feat.append(Feat)

        # # 多尺度特征融合计算
        # MS_spectral_list = []
        # for scale in range(len(MS_spectral)):
        #     spec_vec = MS_spectral[scale].unsqueeze(1)
        #     spec_vec = F.interpolate(spec_vec, size=(l), mode='linear', align_corners=False)
        #     MS_spectral_list.append(spec_vec)

        # MS_spectral_list = torch.cat([x for x in MS_spectral_list], 1)
        # MS_Feat = compute_spatt_conv1D2D(MS_spectral_list, act_list1[scale], act_list2[scale], samesize=samesize)
        # if samesize==True:
        #     MS_Feat = F.interpolate(MS_Feat, size=(l, l), mode='bilinear')
        
        return MS_Feat
def MS_spectral_FeatureExtract_bandwise(MS_spectral, act_list1, act_list2, mlp_list, act='1D', samesize=False):
        '''
        Usage:
                Extract Feature for Multiscale spectral
        Input: 
                MS_spectral [scale1, scale2, scale3] scale1 is original scale 
        Output: List
                MS_Feat(A list, Multiscale features)  [[B, 1, H, W], [B, 1, H/2, W/2]...]
        '''
        b, l = MS_spectral[0].size()

        # 计算多尺度特征
        MS_Feat = []
        for scale in range(len(MS_spectral)):
            spec_vec = MS_spectral[scale].unsqueeze(1)
            if act == 'mlp':
                Feat = compute_bandwise_mlp(spec_vec, act_list1[scale], act_list2[scale], mlp_list[scale], samesize=samesize)
            elif act == '1D':
                Feat = compute_bandwise_conv1D(spec_vec, act_list1[scale], act_list2[scale], mlp_list[scale], samesize=samesize)
            elif act == '2D':
                Feat = compute_bandwise_conv2D(spec_vec, act_list1[scale], act_list2[scale], mlp_list[scale], samesize=samesize)
            else:
                return False
            if samesize==True:
                Feat = F.interpolate(Feat, size=(l, l), mode='bilinear')

            MS_Feat.append(Feat)

        return MS_Feat
def MS_spectral_FeatureExtract_bandwise4D(MS_spectral, act_list1, act_list2, mlp_list, act='2D', samesize=False):
        '''
        Usage:
                Extract Feature for Multiscale spectral Image -> (B, 1, H, W, C, C) 
        Input: 
                MS_spectral Image [scale1, scale2, scale3]   scale1 is original scale ([B, L, H, W])
        Output: List
                MS_Feat(A list, Multiscale features) -> 6D 
                if samesize == True:
                    [[B, 1, H, W, C, C], [B, 1, H, W, C, C]...]
                if samesize == False:
                    [[B, 1, H, W, C, C], [B, 1, H, W, C/2, C/2]...]
        '''
        b, c, h, w = MS_spectral[0].size()

        # 计算多尺度特征
        MS_Feat = []
        for scale in range(len(MS_spectral)):
            Feat = compute_spatial_bandwise(MS_spectral[scale], act_list1[scale], act_list2[scale], mlp_list[scale], act=act, samesize=samesize)
            if samesize==True:
                Feat = F.interpolate(Feat, size=(c, c), mode='bilinear')
            Feat = Feat.view(b, -1, h, w, c, c)
            MS_Feat.append(Feat)

        return MS_Feat
def MS_spectral_FeatureExtract_bandwise4D_WF(MS_spectral, act_list1, act_list2, mlp_list, act='2D', samesize=False):
        '''
        Usage:
                Extract Feature for Multiscale spectral Image -> (B, 1, H, W, C, C) 
        Input: 
                MS_spectral Image [scale1, scale2, scale3]   scale1 is original scale ([B, L, H, W])
        Output: List
                MS_Feat(A list, Multiscale features) -> 6D 
                if samesize == True:
                    [[B, 1, H, W, C, C], [B, 1, H, W, C, C]...]
                if samesize == False:
                    [[B, 1, H, W, C, C], [B, 1, H, W, C/2, C/2]...]
        '''
        b, c, h, w = MS_spectral[0].size()

        # 计算多尺度特征
        MS_Feat = []
        for scale in range(len(MS_spectral)):
            B, C, H, W = MS_spectral[scale].size()
            vect = MS_spectral[scale].view(B, C, -1).permute(0, 2, 1)
            Feat = compute_bandwise_conv2D_WF(vect, act_list1[scale], act_list2[scale], samesize=samesize)
            MS_Feat.append(Feat)

        return MS_Feat
def MS_spectral_FeatureExtract_bandwise4D_LF(MS_spectral, act_list1, act_list2, mlp_list, act='2D', samesize=False):
        '''
        Usage:
                Extract Feature for Multiscale spectral Image -> (B, 1, H, W, C, C) 
        Input: 
                MS_spectral Image [scale1, scale2, scale3]   scale1 is original scale ([B, L, H, W])
        Output: List
                MS_Feat(A list, Multiscale features) -> 6D 
                if samesize == True:
                    [[B, 1, H, W, C, C], [B, 1, H, W, C, C]...]
                if samesize == False:
                    [[B, 1, H, W, C, C], [B, 1, H, W, C/2, C/2]...]
        '''
        b, c, h, w = MS_spectral[0].size()

        # 计算多尺度特征
        MS_Feat = []
        for scale in range(len(MS_spectral)):
            B, C, H, W = MS_spectral[scale].size()
            vect = MS_spectral[scale].view(B, C, -1).permute(0, 2, 1)
            Feat = compute_bandwise_conv2D_LF(vect, act_list1[scale], act_list2[scale], samesize=samesize)
            MS_Feat.append(Feat)

        return MS_Feat
# //////////////////////////////////////////////////////////////
# Compute Band Feature
def compute_covariance_maps(img, scale_T_LIST=[1,2,3,4,5]):
    '''
    Input:  img(B, C, H, W)
            scale_T(The scale para, a 'scale_T' contains 'scale_T*scale_T' pixels)
    '''
    b, l, h, w = img.size()
    cov_map_list = []
    for scale_T in scale_T_LIST:
        T = 2*scale_T + 1
        # 得到光谱图
        pth = int((h - 1)/2)
        ptw = int((w - 1)/2)
        pt_ct = img[:,:,pth,ptw].view(b, -1)

        pt_scale = img[:, :, pth-scale_T:pth+scale_T+1, ptw-scale_T:ptw+scale_T+1].contiguous().view(b, l, -1)
        pt_scale_means = pt_scale.sum(2).view(b, l, -1)
        pt_norm = (pt_scale-pt_scale_means)
        cov_map = torch.bmm(pt_norm, pt_norm.transpose(2,1))
        cov_map_norm = cov_map/(T*T-1)
        cov_map_list.append(cov_map_norm.unsqueeze(1))
    
    cov_maps = torch.cat([x for x in cov_map_list], 1)
    return cov_maps

def compute_pixelwise_relation(vector, mlp):
    b, l = vector.size()
    att_maps = []
    softmax = nn.Softmax(dim=-1)

    vect_squ1 = vector.view(b, 1, l).repeat(1, l, 1).unsqueeze(3)
    vect_squ2 = vector.view(b, l, 1).repeat(1, 1, l).unsqueeze(3)
    vect_pixelwise = torch.cat([vect_squ1, vect_squ2], 3)   # b, l, l, 2
    vect_pixelwise = vect_pixelwise.view(-1, 2)
    mlp_ralation = mlp(vect_pixelwise)    # b * l * l
    # print(mlp_ralation.shape)
    mlp_ralation = mlp_ralation.view(b, -1, l, l)
    return mlp_ralation

def compute_spatt_mlp(vector, mlp1, mlp2):
    b, n, l = vector.size()
    att_maps = []
    softmax = nn.Softmax(dim=-1)
    for img_num in range(n):
        vect_patch = vector[:,img_num,:]
        # print(vect_patch.shape)
        vector_emb1 = mlp1(vect_patch).view(b, -1, l).permute(0, 2, 1)
        vector_emb2 = mlp2(vect_patch).view(b, -1, l)
        attention_s = softmax(torch.bmm(vector_emb1, vector_emb2)).view(b, 1, l, l)
        att_maps.append(attention_s)
    
    att_maps = torch.cat([x for x in att_maps], 1)
    return att_maps

def MultiScale_SP(SP_img, scale_list=[1, 2, 4, 8, 16]):
        '''
        Usage:
                First Compute SP, then compute MultiScale SP
        Input: 
                SP_img(step\MLP\Ratio......, B, 1, H, W) 
        Output:
                SP_MultiScale(A list, SP_img with same position, different scale)  [[B, 1, H, W], [B, 1, H/2, W/2]...]
        '''
        b, _, h, w = SP_img.size()
        
        # 计算5层多尺度级联
        SP_MultiScale = []
        for scale in scale_list:
            SP_scale = F.interpolate(SP_img, size=(int((h)/scale), int((w)/scale)), mode='bilinear')
            SP_MultiScale.append(SP_scale)
        
        return SP_MultiScale

def MultiScale_CT(vector_ct, act_list1, act_list2, scale_list=[1, 2, 4, 8, 16], samesize=False):
        '''
        Usage:
                First Compute MultiScale vector_ct, then compute SP of MultiScale vector_ct
        Input: 
                vector_ct(center pixel of img)
        Output:
                SP_MultiScale(A list, SP_img with same position, different scale)  [[B, 1, H, W], [B, 1, H/2, W/2]...]
        '''
        b, l = vector_ct.size()
        vector_ct = vector_ct.view(b, 1, l)

        # 计算5层多尺度级联
        SP_MultiScale = []
        layer = 0
        for scale in scale_list:
            vector_re = F.interpolate(vector_ct, size=(int(l/scale)), mode='linear', align_corners=False)
            # SP_singlescale = compute_auto_mulply(vector_re, act_list1[layer], act_list2[layer], 0)
            SP_singlescale = compute_spatt_conv2D(vector_re, act_list1[layer], act_list2[layer], samesize=samesize)
            # SP_singlescale = compute_pixelwise_relation(vector_re.view(b, -1), act_list1[layer])
            if samesize==True:
                SP_singlescale = F.interpolate(SP_singlescale, size=(l, l), mode='bilinear')
            # SP_singlescale = compute_ratio_withstep(vector_re.view(b, -1))
            SP_MultiScale.append(SP_singlescale)
            layer += 1
            # print(vector_ct.shape, vector_re.shape, SP_singlescale.shape)
        return SP_MultiScale

def compute_auto_mulply_triple(vector, mlp1, mlp2, mlp3):
    b, l = vector.size()
    att_maps = []
    softmax = nn.Softmax(dim=-1)

    vector_emb1 = mlp1(vector).view(b, -1, l).permute(0, 2, 1)
    vector_emb2 = mlp2(vector).view(b, -1, l)
    attention_s = torch.bmm(vector_emb1, vector_emb2).view(b, -1, l*l).permute(0, 2, 1)
    vector_emb3 = mlp3(vector).view(b, -1, l)
    
    attention_triple = softmax(torch.bmm(attention_s, vector_emb3)).view(b, 1, l, l, l)
    return attention_triple

def compute_auto_mulply(vector, mlp1, mlp2, mode=0):
    b, n, l = vector.size()
    att_maps = []
    softmax = nn.Softmax(dim=-1)
    if mode == 0:           # multiply
        for img_num in range(n):
            vect_patch = vector[:,img_num,:]
            # print(vect_patch.shape, mlp1)
            vector_emb1 = mlp1(vect_patch).view(b, -1, l).permute(0, 2, 1)
            vector_emb2 = mlp2(vect_patch).view(b, -1, l)
            attention_s = softmax(torch.bmm(vector_emb1, vector_emb2)).view(b, 1, l, l)
            att_maps.append(attention_s)
    elif mode == 1:         # step
        for img_num in range(n):
            vect_patch = vector[:,img_num,:]
            vector_emb1 = mlp1(vect_patch).view(b, -1, l)
            attention_s = softmax(compute_ratio_withstep_entireimage(vector_emb1)).view(b, 1, l, l)
            att_maps.append(attention_s)
    elif mode == 2:         # ratio
        for img_num in range(n):
            vect_patch = vector[:,img_num,:]
            vector_emb1 = mlp1(vect_patch).view(b, -1, l).permute(0, 2, 1)
            vector_emb2 = 1 / mlp2(vect_patch).view(b, -1, l)
            attention_s = softmax(torch.bmm(vector_emb1, vector_emb2)).view(b, 1, l, l)
            att_maps.append(attention_s)
    
    att_maps = torch.cat([x for x in att_maps], 1)
    return att_maps

def compute_multi_features_mlp(vector, mlp_list1, mlp_list2):
    b, l = vector.size()
    att_maps = []
    softmax = nn.Softmax(dim=-1)
    
    # multiply
    vector_emb1 = mlp_list1[0](vector).view(b, -1, l).permute(0, 2, 1)
    vector_emb2 = mlp_list2[0](vector).view(b, -1, l)
    attention_s = softmax(torch.bmm(vector_emb1, vector_emb2)).view(b, 1, l, l)
    att_maps.append(attention_s)

    # # ratio
    # vector_emb1 = mlp_list1[1](vector).view(b, -1, l).permute(0, 2, 1)
    # vector_emb2 = 1 / mlp_list2[1](vector).view(b, -1, l)
    # attention_s = softmax(torch.bmm(vector_emb1, vector_emb2)).view(b, 1, l, l)
    # att_maps.append(attention_s)

    att_maps = torch.cat([x for x in att_maps], 1)
    return att_maps

def compute_ratio_mlp(vector, mlp1, mlp2):
    b, n, l = vector.size()
    att_maps = []
    softmax = nn.Softmax(dim=-1)
    for img_num in range(n):
        vect_patch = vector[:,img_num,:]
        # print(vect_patch.shape)
        vector_emb1 = mlp1(vect_patch).view(b, -1, l).permute(0, 2, 1)
        vector_emb2 = 1 / (mlp2(vect_patch).view(b, -1, l))
        attention_s = softmax(torch.bmm(vector_emb1, vector_emb2)).view(b, 1, l, l)
        att_maps.append(attention_s)
    
    att_maps = torch.cat([x for x in att_maps], 1)
    return att_maps

def compute_ratio_withstep(vector):
    b, l = vector.size()

    # 处理0值，将其处理为非0值里面的最小值
    zero_index = (vector == 0)
    nonzero_index = (vector!=0)
    if nonzero_index.sum() != 0:
        vector[zero_index] = torch.min(vector[nonzero_index])
    else:
        vector[zero_index] = 1

    vect_squ1 = vector.view(b, 1, l).repeat(1, l, 1)
    vect_squ2 = vector.view(b, l, 1).repeat(1, 1, l)

    # step1 - step2之后，步长对角线的值为0，做除数会出错，因此再生成一个单位矩阵，加到step1 - step2之后
    step1 = torch.tensor([i for i in range(l)]).view(1, 1, l).repeat(b, l, 1)
    step2 = torch.tensor([i for i in range(l)]).view(1, l, 1).repeat(b, 1, l)
    step_diag = torch.eye(l).view(1, l, l).repeat(b, 1, 1)
    step = (step1 - step2 + step_diag).cuda()
    return torch.abs((vect_squ1 - vect_squ2)/step).unsqueeze(1)
    # return ((vect_squ1 - vect_squ2)/step).unsqueeze(1)

def compute_ratio(vector):
    b, l = vector.size()

    # 处理0值，将其处理为非0值里面的最小值
    zero_index = (vector == 0)
    nonzero_index = (vector!=0)
    vector[zero_index] = torch.min(vector[nonzero_index])

    vect_squ1 = vector.view(b, 1, l).repeat(1, l, 1)
    vect_squ2 = vector.view(b, l, 1).repeat(1, 1, l)

    return (vect_squ1 / vect_squ2).unsqueeze(1)

def compute_ratio_withstep_entireimage(vector_in):
    b, n, l = vector_in.size()
    vector = vector_in.clone()

    # 处理0值，将其处理为非0值里面的最小值
    zero_index = (vector == 0)
    nonzero_index = (vector!=0)
    vector[zero_index] = torch.min(vector[nonzero_index])

    vect_squ1 = vector.view(b, n, 1, l).repeat(1, 1, l, 1)
    vect_squ2 = vector.view(b, n, l, 1).repeat(1, 1, 1, l)

    # step1 - step2之后，步长对角线的值为0，做除数会出错，因此再生成一个单位矩阵，加到step1 - step2之后
    step1 = torch.tensor([i for i in range(l)]).view(1, 1, 1, l).repeat(b, n, l, 1)
    step2 = torch.tensor([i for i in range(l)]).view(1, 1, l, 1).repeat(b, n, 1, l)
    step_diag = torch.eye(l).view(1, 1, l, l).repeat(b, n, 1, 1)
    step = (step1 - step2 + step_diag).cuda()
    return torch.abs((vect_squ1 - vect_squ2)/step)
    
def compute_index(vector):
    b, l = vector.size()

    # 处理0值，将其处理为非0值里面的最小值
    zero_index = (vector == 0)
    nonzero_index = (vector!=0)
    vector[zero_index] = torch.min(vector[nonzero_index])

    vect_squ1 = vector.view(b, 1, l).repeat(1, l, 1)
    vect_squ2 = vector.view(b, l, 1).repeat(1, 1, l)

    return torch.abs(((vect_squ1-vect_squ2) / (vect_squ1 + vect_squ2))).unsqueeze(1)

def compute_spatt_conv1D(vector, conv1D_1, conv1D_2, samesize=False):
    b, n, l = vector.size()
    att_maps = []
    for img_num in range(n):
        vect_patch = vector[:,img_num,:]
        vector_emb1 = conv1D_1(vect_patch.view(b, -1, l)).permute(0, 2, 1)
        vector_emb2 = conv1D_2(vect_patch.view(b, -1, l))
        softmax = nn.Softmax(dim=-1)
        attention_s = softmax(torch.bmm(vector_emb1, vector_emb2)).view(b, 1, l, l)
        att_maps.append(attention_s)
    
    att_maps = torch.cat([x for x in att_maps], 1)
    return att_maps

def compute_spatt_conv2D(vector, conv2D_1, conv2D_2, samesize=False):
    '''
        Input: vector (b, n, l)
    '''
    # print(vector.shape)
    b, n, l = vector.size()

    att_maps = []
    for img_num in range(n):
        vect_patch = vector[:,img_num,:]
        # print(vect_patch.shape)
        vect_2D = compute_local_feature(vect_patch, samesize=samesize)
        # print(vect_2D.shape, vect_patch.shape, conv2D_1(vect_2D).shape)
        vector_emb1 = conv2D_1(vect_2D).view(b, -1, vect_2D.shape[2]*vect_2D.shape[3]).permute(0, 2, 1)
        vector_emb2 = conv2D_2(vect_2D).view(b, -1, vect_2D.shape[2]*vect_2D.shape[3])
        # print(vect_2D.shape, vect_patch.shape, vector_emb1.shape, vector_emb2.shape)
        softmax = nn.Softmax(dim=-1)
        attention_s = softmax(torch.bmm(vector_emb1, vector_emb2)).view(b, 1, vector_emb1.shape[1], vector_emb1.shape[1])
        # print(softmax(torch.bmm(vector_emb1, vector_emb2)).shape, attention_s.shape)
        att_maps.append(attention_s)
    
    att_maps = torch.cat([x for x in att_maps], 1)
    return att_maps

def compute_spatt_conv1D2D_batch(vector, conv1D, conv2D, samesize=False):
    '''
        Input: vector (b, n, l)
    '''
    b, n, l = vector.size()

    softmax = nn.Softmax(dim=-1)

    vect_imglist = []
    for img_num in range(n):
        vect_patch = vector[:,img_num,:]
        vect_2D = compute_local_feature(vect_patch, samesize=samesize)
        vect_imglist.append(vect_2D)
    vect_img = torch.cat([x for x in vect_imglist], 1)

    vector_emb1 = conv2D(vect_img).view(b, -1, vect_img.shape[2]*vect_img.shape[3])
    vector_emb1 = F.interpolate(vector_emb1, size=(l), mode='linear', align_corners=False)
    vector_emb1 = vector_emb1.permute(0, 2, 1)
    vector_emb2 = conv1D(vector.view(b, -1, l))
    
    attention_s = softmax(torch.bmm(vector_emb1, vector_emb2)).view(b, 1, vector_emb1.shape[1], vector_emb1.shape[1])
    return attention_s

def compute_spatt_conv1D2D(vector, conv1D, conv2D, samesize=False):
    '''
        Input: vector (b, n, l)
    '''
    b, n, l = vector.size()
    att_maps = []
    softmax = nn.Softmax(dim=-1)

    for img_num in range(n):
        vect_patch = vector[:,img_num,:]
        vect_2D = compute_local_feature(vect_patch, samesize=samesize)
        vector_emb1 = conv2D(vect_2D).view(b, -1, vect_2D.shape[2]*vect_2D.shape[3])
        vector_emb1 = F.interpolate(vector_emb1, size=(l), mode='linear', align_corners=False)
        vector_emb1 = vector_emb1.permute(0, 2, 1)
        vector_emb2 = conv1D(vect_patch.view(b, -1, l))
        
        attention_s = softmax(torch.bmm(vector_emb1, vector_emb2)).view(b, 1, vector_emb1.shape[1], vector_emb1.shape[1])
        att_maps.append(attention_s)
    
    att_maps = torch.cat([x for x in att_maps], 1)
    return att_maps

def compute_local_feature_batch(pt_ct, samesize=True):
    '''
        Input: center pixel([b, n, l])
        Out: local features(pt reshape into an image)(b*n, 1, imgsize, imgsize)
    '''
    b, n, l = pt_ct.size()
    pt_ct = pt_ct.contiguous().view(b*n, l)

    imgsize = 15

    pt_ct = F.interpolate(pt_ct.view(-1, 1, l), size=(imgsize*imgsize), mode='linear', align_corners=False)
    pt_ct = pt_ct.view(-1, 1, imgsize, imgsize)
    return pt_ct

def compute_local_feature(pt_ct, samesize=False):
    '''
        Input: center pixel([b, l])
        Out: local features(pt reshape into an image)(b, 1, imgsize, imgsize)
    '''
    b, l,  = pt_ct.size()
    if samesize == True:
        imgsize = 14
    else: 
        if l == 103:
            imgsize = 50
        elif l == 144:
            imgsize = 12
        elif l == 48:
            imgsize = 7
        elif l == 204:
            imgsize = 14

    pt_ct = F.interpolate(pt_ct.view(b, 1, l), size=(imgsize*imgsize), mode='linear', align_corners=False)
    pt_ct = pt_ct.view(b, 1, imgsize, imgsize)
    return pt_ct
# //////////////////////////////////////////////////////////////
# Compute bandwise
def compute_bandwise_conv2D(vector, conv2D_1, conv2D_2, mlp, samesize=False):
    '''
        Input: vector (b, n, l)
    '''
    b, n, l = vector.size()
    att_maps = []
    softmax = nn.Softmax(dim=-1)
    vect_2D_list = []

    vect_2D_list = compute_local_feature_batch(vector, samesize=samesize)
    vector_emb1 = conv2D_1(vect_2D_list).view(b*n, -1, vect_2D_list.shape[2]*vect_2D_list.shape[3])
    # vector_emb1 = vect_2D_list.view(b*n, -1, vect_2D_list.shape[2]*vect_2D_list.shape[3])
    vector_emb1 = F.interpolate(vector_emb1, size=(l), mode='linear', align_corners=False)
    
    vector_emb2 = conv2D_2(vect_2D_list).view(b*n, -1, vect_2D_list.shape[2]*vect_2D_list.shape[3])
    # vector_emb2 = vect_2D_list.view(b*n, -1, vect_2D_list.shape[2]*vect_2D_list.shape[3])
    vector_emb2 = F.interpolate(vector_emb2, size=(l), mode='linear', align_corners=False)

    vector_emb2 = vector_emb2.view(b*n, -1, l, 1).repeat(1, 1, 1, l)
    vector_emb1 = vector_emb1.view(b*n, -1, 1, l).repeat(1, 1, l, 1)
    vect_pixelwise = torch.cat([vector_emb1, vector_emb2], 1)   # b*n, 2*ratio, l, l

    vect_pixelwise = vect_pixelwise.view(b*n, -1, l*l).permute(0, 2, 1).contiguous().view(b*l*l*n, -1)

    band_ralation = mlp(vect_pixelwise)    # b * l * l
    band_ralation = band_ralation.view(b, -1, l, l)
    att_maps.append(band_ralation)

    att_maps = torch.cat([x for x in att_maps], 1)
    
    return att_maps
    # vect_emb1 = conv2D_1(vect_2D_list)
    # vect_emb2 = conv2D_2(vect_2D_list)
    # return torch.cat([vect_emb1, vect_emb2], 1)

def compute_bandwise_conv2D_LF(vector, conv2D_1, conv2D_2, samesize=False):
    '''
        Input: vector (b, n, l)
    '''
    b, n, l = vector.size()
    att_maps = []
    softmax = nn.Softmax(dim=-1)
    vect_2D_list = []

    vect_2D_list = compute_local_feature_batch(vector, samesize=samesize)
    vector_emb1 = conv2D_1(vect_2D_list)
    vector_emb2 = conv2D_2(vect_2D_list)
    
    vect_emb = torch.cat([vector_emb1, vector_emb2], 1).view(b, n, -1, vector_emb1.shape[2], vector_emb1.shape[3])
    return vect_emb

def compute_bandwise_conv2D_old(vector, conv2D_1, conv2D_2, mlp, samesize=False):
    '''
        Input: vector (b, n, l) 速度慢
    '''
    b, n, l = vector.size()
    att_maps = []
    softmax = nn.Softmax(dim=-1)
    vect_pixelwise_list = []
    for img_num in range(n):
        vect_patch = vector[:,img_num,:]
        vect_2D = compute_local_feature(vect_patch, samesize=samesize)
        vector_emb1 = conv2D_1(vect_2D).view(b, -1, vect_2D.shape[2]*vect_2D.shape[3])
        vector_emb1 = F.interpolate(vector_emb1, size=(l), mode='linear', align_corners=False)

        vector_emb2 = conv2D_2(vect_2D).view(b, -1, vect_2D.shape[2]*vect_2D.shape[3])
        vector_emb2 = F.interpolate(vector_emb2, size=(l), mode='linear', align_corners=False)

        vector_emb2 = vector_emb2.view(b, -1, l, 1).repeat(1, 1, 1, l)
        vector_emb1 = vector_emb1.view(b, -1, 1, l).repeat(1, 1, l, 1)
        vect_pixelwise = torch.cat([vector_emb1, vector_emb2], 1)   # b, 2*ratio, l, l
        vect_pixelwise = vect_pixelwise.view(b, -1, l*l).permute(0, 2, 1).contiguous().view(b*l*l, -1)
        vect_pixelwise_list.append(vect_pixelwise)
    
    vect_pixelwise_list = torch.cat([x for x in vect_pixelwise_list], 0)
    band_ralation = mlp(vect_pixelwise_list)    # b * l * l
    band_ralation = band_ralation.view(b, -1, l, l)
    att_maps.append(band_ralation)
    
    att_maps = torch.cat([x for x in att_maps], 1)
    return att_maps
    
def compute_bandwise_conv1D2D(vector, conv1D, conv2D, mlp, samesize=False):
    '''
        Input: vector (b, n, l)
    '''
    b, n, l = vector.size()
    att_maps = []
    softmax = nn.Softmax(dim=-1)

    for img_num in range(n):
        vect_patch = vector[:,img_num,:]
        vect_2D = compute_local_feature(vect_patch, samesize=samesize)
        vector_emb1 = conv2D(vect_2D).view(b, -1, vect_2D.shape[2]*vect_2D.shape[3])
        vector_emb1 = F.interpolate(vector_emb1, size=(l), mode='linear', align_corners=False)
        vector_emb1 = vector_emb1.permute(0, 2, 1)
        vector_emb2 = conv1D(vect_patch.view(b, -1, l))
        
        vector_emb1 = vector_emb1.view(b, -1, l).repeat(1, l, 1).unsqueeze(3)
        vector_emb2 = vector_emb2.view(b, l, -1).repeat(1, 1, l).unsqueeze(3)
        vect_pixelwise = torch.cat([vector_emb1, vector_emb2], 3)   # b, l, l, 2
        vect_pixelwise = vect_pixelwise.view(-1, 2)
        band_ralation = mlp(vect_pixelwise)    # b * l * l
        band_ralation = band_ralation.view(b, -1, l, l)
        att_maps.append(band_ralation)
    
    att_maps = torch.cat([x for x in att_maps], 1)
    return att_maps
def compute_bandwise_mlp(vector, mlp1, mlp2, mlp_act, samesize=False):
    b, n, l = vector.size()
    att_maps = []
    softmax = nn.Softmax(dim=-1)

    for img_num in range(n):
        vect_patch = vector[:,img_num,:]
        vector_emb1 = mlp1(vect_patch)
        vector_emb2 = mlp2(vect_patch)

        vector_emb2 = vector_emb2.view(b, -1, l, 1).repeat(1, 1, 1, l)
        vector_emb1 = vector_emb1.view(b, -1, 1, l).repeat(1, 1, l, 1)
        # print(vector_emb1, vector_emb2)
        vect_pixelwise = torch.cat([vector_emb1, vector_emb2], 1)   # b, l, l, 2
        vect_pixelwise = vect_pixelwise.view(b, -1, l*l).permute(0, 2, 1).contiguous().view(b*l*l, -1)

        band_ralation = mlp_act(vect_pixelwise)    # b * l * l
        band_ralation = band_ralation.view(b, -1, l, l)
        att_maps.append(band_ralation)
    
    att_maps = torch.cat([x for x in att_maps], 1)
    return att_maps

def compute_bandwise_conv1D(vector, conv1D_1, conv1D_2, mlp, samesize=False):
    '''
        Input: vector (b, n, l)
    '''
    b, n, l = vector.size()
    att_maps = []
    vector = vector.contiguous().view(b*n, 1, l)

    vector_emb1 = conv1D_1(vector).view(b*n, -1, l)
    # vector_emb1 = F.interpolate(vector_emb1, size=(l), mode='linear', align_corners=False)
    
    vector_emb2 = conv1D_2(vector).view(b*n, -1, l)
    # vector_emb2 = vect_2D_list.view(b*n, -1, vect_2D_list.shape[2]*vect_2D_list.shape[3])
    # vector_emb2 = F.interpolate(vector_emb2, size=(l), mode='linear', align_corners=False)

    vector_emb2 = vector_emb2.view(b*n, -1, l, 1).repeat(1, 1, 1, l)
    vector_emb1 = vector_emb1.view(b*n, -1, 1, l).repeat(1, 1, l, 1)
    vect_pixelwise = torch.cat([vector_emb1, vector_emb2], 1)   # b*n, 2*ratio, l, l

    vect_pixelwise = vect_pixelwise.view(b*n, -1, l*l).permute(0, 2, 1).contiguous().view(b*l*l*n, -1)

    band_ralation = mlp(vect_pixelwise)    # b * l * l
    band_ralation = band_ralation.view(b, -1, l, l)
    att_maps.append(band_ralation)

    att_maps = torch.cat([x for x in att_maps], 1)
    
    return att_maps

def compute_spatial_bandwise(Hyper_Image, act_1, act_2, mlp, act = '2D', samesize=True):
    '''
        Input:  Image(Hyperspectral image) -> (B, C, H, W)
                Conv2D_1: Activate Conv2D 1
                Conv2D_2: Activate Conv2D 2
                mlp: Bandwise mlp
        Output:
                4D Spatial & Bandwise image -> (B, H*W, h, w) (B: Batch, H: image H, W: image W, h: spectral image (C), w: spectral image (C))
    '''
    B, C, H, W = Hyper_Image.size()
    vect = Hyper_Image.view(B, C, -1).permute(0, 2, 1)
    if act == 'mlp':
        spa_bandwise = compute_bandwise_mlp(vect, act_1, act_2, mlp, samesize=samesize)     # B, H*W, C, C
    elif act == '1D':
        spa_bandwise = compute_bandwise_conv1D(vect, act_1, act_2, mlp, samesize=samesize)  # B, H*W, C, C
    elif act == '2D':
        spa_bandwise = compute_bandwise_conv2D(vect, act_1, act_2, mlp, samesize=samesize)  # B, H*W, C, C
    else:
        return False
    
    return spa_bandwise

# //////////////////////////////////////////////////////////////
# Compute Band Weights
def Spectral_Weighted_MulChannel(Feats_3D, ratio_img):
    torch.cuda.empty_cache()
    b, c, l, h, w = Feats_3D.size()
    
    Feats_3D_Ori = Feats_3D
    softmax = nn.Softmax(dim=-1)
    att_map = softmax(ratio_img)

    Feats_3D = Feats_3D.view(b, c, l, h*w).permute(0, 1, 3, 2)
    att_map = att_map.view(b, c, l, l)
    result = torch.matmul(Feats_3D, att_map).permute(0, 1, 3, 2).view(b, c, l, h, w)

    return result + Feats_3D_Ori

# //////////////////////////////////////////////////////////////
# Pooling operations
def local_pool(feat_map):
    b, c, h, w = feat_map.size()

    lu = feat_map[:,:,0:int(h/2),0:int(w/2)].reshape(b, c, -1)
    ld = feat_map[:,:,int(h/2):h,0:int(w/2)].reshape(b, c, -1)
    ru = feat_map[:,:,0:int(h/2),int(w/2):w].reshape(b, c, -1)
    rd = feat_map[:,:,int(h/2):h,int(w/2):h].reshape(b, c, -1)
    # print(lu.shape, ld.shape, ru.shape, rd.shape)

    lu_mean = torch.mean(lu, 2).view(b, c, 1, 1)
    ld_mean = torch.mean(ld, 2).view(b, c, 1, 1)
    ru_mean = torch.mean(ru, 2).view(b, c, 1, 1)
    rd_mean = torch.mean(rd, 2).view(b, c, 1, 1)
    # print(lu_mean.shape, ld_mean.shape, ru_mean.shape, rd_mean.shape)

    pool_img_l = torch.cat([lu_mean, ld_mean], 2)
    pool_img_r = torch.cat([ru_mean, rd_mean], 2)
    pool_img = torch.cat([pool_img_l, pool_img_r], 3)
    # print(pool_img.shape)

    return pool_img

def local_linepool(feat_map):
    b, c, h, w = feat_map.size()

    m1, m2, m3, m4 = int(h*w/4), int(2*h*w/4), int(3*h*w/4), int(4*h*w/4)

    feat_map_flat = feat_map.view(b, c, -1)

    mean1 = torch.mean(feat_map_flat[:,:,0:m1], 2).view(b, c, 1, 1)
    mean2 = torch.mean(feat_map_flat[:,:,m1:m2], 2).view(b, c, 1, 1)
    mean3 = torch.mean(feat_map_flat[:,:,m2:m3], 2).view(b, c, 1, 1)
    mean4 = torch.mean(feat_map_flat[:,:,m3:m4], 2).view(b, c, 1, 1)

    return torch.cat([mean1, mean2, mean3, mean4], 3)
