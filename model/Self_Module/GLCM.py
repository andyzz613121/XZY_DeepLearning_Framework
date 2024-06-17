import numpy as np
from PIL import Image
import skimage.feature as feature
from sklearn.metrics.cluster import entropy

def compute_glcm(img, distance=[1], direction=[0, np.pi/4, np.pi/2, 3*np.pi/4], gray_level=8, props={'contrast', 'dissimilarity','homogeneity', 'energy', 'correlation', 'ASM', 'entropy'}):
    img = quantize_image(img, level=gray_level)
    glcm = feature.greycomatrix(img, distance, direction, levels=gray_level)
    print(glcm.shape)
    values_temp = {}
    for prop in props:
        if prop == 'entropy':
            temp = 0
            for dist in range(len(distance)):
                for dirt in range(len(direction)):
                    print(glcm[:,:,dist,dirt])
                    print(entropy(glcm[:,:,dist,dirt]))
                    temp += entropy(glcm[:,:,dist,dirt])
        else:
            temp = feature.greycoprops(glcm, prop)
        values_temp[prop] = temp

    return values_temp

def quantize_image(img, level=8, max_value=256):
    '''
        Input: img[h, w]
               level(量化级)
               max_value(灰度范围)
    '''
    gray_range = [ int(256*x/level) for x in range(0, level+1) ]
    # gray_index = [x for x in range(0, level)]
    quantize_img = np.zeros_like(img) - 100

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pix_value = img[i][j]
            for gray_value in range(0, level):
                if (pix_value >= gray_range[gray_value] and pix_value <= gray_range[gray_value+1]):
                    quantize_img[i][j] = gray_value
    return quantize_img

if __name__ == '__main__':
    from PIL import Image
    img_file = 'F:\\Project\\ImageForgery\\沈帆\模糊篡改目标识别\\数据集\\Fuzzy Data\\Ori\\High Density\\HD (1).png'
    img = np.array(Image.open(img_file))
    img = img[:, :, 0]*0.84 + img[:, :, 1]*0.11 + img[:, :, 2]*0.05
    img = img.astype(np.uint8)
    g = compute_glcm(img, gray_level=4)
    print(g)
    # img = quantize_image(img)
    # img = Image.fromarray(img)
    # img.save('F:\\Project\\ImageForgery\\沈帆\模糊篡改目标识别\\数据集\\Fuzzy Data\\Ori\\High Density\\HD (1)11.png')