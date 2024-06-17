import numpy as np
import cv2

def JMD(img, label, c1, c2):
    c1_index = (label==c1)
    c2_index = (label==c2)
    c1_mean = np.mean(img[c1_index])
    c2_mean = np.mean(img[c2_index])
    c1_std = np.std(img[c1_index])
    c2_std = np.std(img[c2_index])
    a = (1/8) * (c1_mean-c2_mean) * (1/((c1_std+c2_std)/2)) * (c1_mean-c2_mean) + (1/2) * np.log((c1_std+c2_std)/(2*np.sqrt(np.abs(c1_std*c2_std))))
    JM = 2*(1-np.exp(-1*a))
    print(c1_mean, c2_mean, c1_std, c2_std, JM)
    return


if __name__ == "__main__":
    img = cv2.imread('C:\\Users\\25321\\Desktop\\1.tif')
    img[50:100,50:100] = 128
    img[50:100,0:50] = 64
    cv2.imwrite('C:\\Users\\25321\\Desktop\\1.tif',img)
    JMD(img, img, 0, 127)
