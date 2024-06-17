from PIL import Image
from PIL import ImageFilter
import numpy as np
def GaussianBlur(img, r=1):
    '''
        input: img(H*W)  numpy
    '''
    img=Image.fromarray(img)
    img=img.filter(ImageFilter.GaussianBlur(radius=r))
    return np.array(img)
    # img.show()
    

if __name__ == '__main__':
    import os
    for r in [1.2, 1.4, 1.6, 1.8]:
        basefolder = 'E:\\dataset\\ImageBlur\\Data\\train\\实验图像_(复杂度+模糊半径)\\选择的实验图像\\全部\\r0\\High IC\\img_rgb\\'
        outfolder = 'E:\\dataset\\ImageBlur\\Data\\train\\实验图像_(复杂度+模糊半径)\\选择的实验图像\\全部\\r' + str(r) + '\\High IC\\img_rgb\\'
        if os.path.exists(outfolder) == False:
            os.makedirs(outfolder)
        for item in os.listdir(basefolder):
            img = np.array(Image.open(basefolder+item))
            img_Guss = GaussianBlur(img, r)
            img[0:128,0:128] = img_Guss[0:128,0:128]
            img = Image.fromarray(img)
            img.save(outfolder+item)
    # lab = 'E:\\dataset\\ImageBlur\\Data\\train\\无模糊图像（复杂度+模糊半径）\\selected\\r_5\\train\\lab\\'
    # l = np.zeros([256,256]).astype(np.uint8)
    # l[0:128,0:128] = 1
    # l = Image.fromarray(l)
    # l.save(lab+'lab.png')