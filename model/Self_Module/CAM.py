from mimetypes import init
import cv2
import torch
import numpy as np
from PIL import Image
from matplotlib import cm
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

class CAM():
    '''
    Usage: 
    1. 'net.layer4.register_forward_hook(forward_hook)    # 对net.layer4这一层注册前向传播', 特征图存在self.feats_list中
    2. compute_CAM()

    只能用于GAP+fc(n, out_channel)，即一层fc的时候
    '''
    def __init__(self):
        self.init()

    def init(self):
        self.feats_list = []

    
    def forward_hook(self, module, inp, outp):
        '''
        定义hook
        '''
        self.init()
        self.feats_list.append(outp)    # 把输出装入字典feature_map
    
    def get_layer(self, net, name):
        '''
        inputs
            net: The net model
            name: The layer name of 'fc'

        return
            layer: The layer of name
        '''
        return net._modules.get(name)

    def get_weight(self, out, layer, index):
        '''
        inputs
            out: The predict maps
            layer: The net layer
            index: The index of layer (for layer that is Sequential)

        return
            weights: The weights of fc
        '''
        # cls = torch.argmax(out).item()    # 获取预测类别编码
        cls = torch.argmax(out, 1)    # 获取预测类别编码，每个类都有不同的类激活图，通过修改cls的数量来调整选择哪一组fc权重
        print(cls.shape)
        print(layer[index].weight.data.shape, cls)
        return layer[index].weight.data[cls,:]
    
    def _normalize(self, cams):
        """CAM normalization"""
        cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
        cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))

        return cams

    def compute_CAM(self, net, out_maps, layer_name, index):
        '''
        inputs
            net: The net model
            out_maps: The predict maps
            layer_name: The layer name of 'fc'
            index: The index of layer (for layer that is Sequential)

        return
            CAM map
        '''
        if len(self.feats_list) == 0:
            print('self.feats_list中 is None, please using first: net.layerXX.register_forward_hook(forward_hook)')
            return False
        layer = self.get_layer(net, layer_name)
        weights = self.get_weight(out_maps, layer, index)
        # print(weights.shape, weights.view(*weights.shape, 1, 1).shape,  self.feats_list中[0].shape,  self.feats_list中[0].squeeze(0).shape)
        
        cam = (weights.view(*weights.shape, 1, 1) * self.feats_list[0].squeeze(0)).sum(0)
        cam = self._normalize(F.relu(cam, inplace=True)).cpu()
        mask = to_pil_image(cam.detach().numpy(), mode='F')
        return cam, mask


class Grad_CAM():
    '''
        Usage: 
        1. 'net.layerXXX.register_forward_hook(forward_hook)    # 对net.layerXXX这一层注册前向传播', 特征图存在self.feats_list中
        2. 'net.layerXXX.register_backward_hook(backward_hook)    
        3. Compute_GradCAM
    '''
    def __init__(self):
        self.init()

    def init(self):
        self.grads_list = []
        self.feats_list = []

    def backward_hook(self, module, grad_in, grad_out):
        '''
        定义hook
        '''
        # self.init()
        self.grads_list.append(grad_out[0].detach())
        # print(len(self.grads_list))
    def forward_hook(self, module, input, output):
        '''
        定义hook
        '''
        # self.init()
        self.feats_list.append(output)

    def compute_GradCAM(self):
        """
        依据梯度和特征图，生成cam
        :param feature_map: np.array， in [C, H, W]
        :param grads: np.array， in [C, H, W]
        :return: np.array, [H, W]
        """
        # print(self.grads_list)
        feats = self.feats_list[0][0].cpu().detach().numpy()
        grads = self.grads_list[0][0].cpu().detach().numpy()
        cam = np.zeros(feats.shape[1:], dtype=np.float32)  # cam shape (H, W)
        print(cam.shape, grads.shape, feats.shape)
        weights = np.mean(grads, axis=(1, 2))  #
        
        for i, w in enumerate(weights):
            cam += w * feats[i, :, :]
            print(cam.shape, grads.shape, feats.shape, weights.shape, w, i)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (32, 32))
        cam -= np.min(cam)
        cam /= np.max(cam)

        self.init()
        return cam
    
    def compute_GradCAM_Batch(self):
        """
        依据梯度和特征图，生成cam
        :param feature_map: np.array， in [B, C, H, W]
        :param grads: np.array， in [B, C, H, W]
        :return: np.array, [B, H, W]
        """
        feats = self.feats_list[0].cpu().detach().numpy()
        grads = self.grads_list[0].cpu().detach().numpy()
        b, c ,h, w = feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3]

        cam = np.zeros([b, h, w], dtype=np.float32)  # cam shape (H, W)
        weights = np.mean(grads, axis=(2, 3))  #
        weights = weights.reshape((b, c, 1, 1)).repeat(h, axis=2).repeat(w, axis=3)
        cam = weights*feats
        cam = cam.sum(1)
        # print(cam.shape)
        # for i, w in enumerate(weights):
            
        #     cam += w * feats[:, i, :, :]
        # print(cam.shape, grads.shape, feats.shape, weights.shape)
        cam = np.maximum(cam, 0)
        cam_list = []
        for item in range(b):
            cam_b = cv2.resize(cam[item], (100, 100))
            # print(cam.shape)
            cam_b -= np.min(cam_b)
            cam_b /= np.max(cam_b)
            cam_list.append(cam_b)
        cam = np.array(cam_list)
        self.init()
        return cam


def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.6) -> Image.Image:
    """Overlay a colormapped mask on a background image

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image
    
    Usage:
        'result = overlay_mask(orign_img, heatmap) '
        'result.show()' OR
        'result.save(save_path)'
    """


    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError('img and mask arguments need to be PIL.Image')

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError('alpha argument is expected to be of type float between 0 and 1')

    cmap = cm.get_cmap(colormap)    
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, 1:]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img