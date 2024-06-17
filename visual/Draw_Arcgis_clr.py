import sys
base_path = '..\\XZY_DeepLearning_Framework\\'
sys.path.append(base_path)

from data_processing.Raster import *
def read_clr(file_path):
    r_list, g_list, b_list = [], [], []
    with open(file_path) as f:
        for line in f.readlines():
            _, r, g, b = line.split(' ')
            r_list.append(r)
            g_list.append(g)
            b_list.append(b)
    return r_list, g_list, b_list

def draw_img_with_clrlist(img_infile, img_outfile, r_list, g_list, b_list):
    img, para = gdal_read_tif(img_infile)
    imgout = np.zeros([3, para[0], para[1]]).astype(np.uint8)
    img_max, img_min = np.max(img), np.min(img)

    assert len(r_list) == (img_max-img_min+1), 'Color number != Class number'

    for idx in range(len(r_list)):
        pos_idx = (img==(idx+img_min))
        imgout[0][pos_idx], imgout[1][pos_idx], imgout[2][pos_idx] = r_list[idx], g_list[idx], b_list[idx]

    from PIL import Image
    # gdal_write_tif(img_outfile, imgout, para[0], para[1], 3, para[3], para[4])
    imgout = Image.fromarray(imgout.transpose(1,2,0))
    imgout.save(img_outfile)
    return True

def draw_img_with_arcgisclrlist(img_infile, img_outfile, arcgisclrfile):
    r_list, g_list, b_list = read_clr(arcgisclrfile)
    draw_img_with_clrlist(img_infile, img_outfile, r_list, g_list, b_list)
    return True


if __name__ == '__main__':
    # import os
    # for date in ['10_17', '9_12', '8_04', '7_14', '6_29', '5_20']:
    #     if date == '10_17':
    #         imgpath = 'E:\\dataset\\毕设数据\\new\\2. MS\\S2B_MSIL2A_20221017T164259_N0400_R126_T16SBF_20221017T205409.SAFE\\GRANULE\\L2A_T16SBF_A029324_20221017T164640\\IMG_DATA\\R10m\\T16SBF_20221017T164259_TCI_10m.jp2'
    #     elif date == '9_12':
    #         imgpath = 'E:\\dataset\\毕设数据\\new\\2. MS\\S2A_MSIL2A_20220912T163911_N0400_R126_T16SBF_20220913T004803.SAFE\\GRANULE\\L2A_T16SBF_A037732_20220912T164408\\IMG_DATA\\R10m\\T16SBF_20220912T163911_TCI_10m.jp2'
    #     elif date == '8_04':
    #         imgpath = 'E:\\dataset\\毕设数据\\new\\2. MS\\S2A_MSIL2A_20220803T163911_N0400_R126_T16SBF_20220804T003407.SAFE\\GRANULE\\L2A_T16SBF_A037160_20220803T164737\\IMG_DATA\\R10m\\T16SBF_20220803T163911_TCI_10m.jp2'
    #     elif date == '7_14':
    #         imgpath = 'E:\\dataset\\毕设数据\\new\\2. MS\\S2A_MSIL2A_20220724T163851_N0400_R126_T16SBF_20220725T004007.SAFE\\GRANULE\\L2A_T16SBF_A037017_20220724T164409\\IMG_DATA\\R10m\\T16SBF_20220724T163851_TCI_10m.jp2'
    #     elif date == '6_29':
    #         imgpath = 'E:\\dataset\\毕设数据\\new\\2. MS\\S2B_MSIL2A_20220629T163839_N0400_R126_T16SBF_20220629T193559.SAFE\\GRANULE\\L2A_T16SBF_A027751_20220629T164910\\IMG_DATA\\R10m\\T16SBF_20220629T163839_TCI_10m.jp2'
    #     elif date == '5_20':
    #         imgpath = 'E:\\dataset\\毕设数据\\new\\2. MS\\S2B_MSIL2A_20220520T163839_N0400_R126_T16SBF_20220520T195256.SAFE\\GRANULE\\L2A_T16SBF_A027179_20220520T165147\\IMG_DATA\\R10m\\T16SBF_20220520T163839_TCI_10m.jp2'
    #     imgpath = 'D:\\毕业\\博士论文\\毕业论文\\图\\第四章\\实验结果\\结果图\\Sentinel'+date+'_LGSF.png'
    #     # imgpath = 'D:\\毕业\\博士论文\\毕业论文\\图\\第二章\\年度数据集图像\\labels_with_year.tif'
    #     img, para = gdal_read_tif(imgpath)
    #     img = img[:, 9015:9230, 4420:4635]
    #     imgoutpath = 'D:\\毕业\\博士论文\\毕业论文\\图\\第五章\\流程图\\SS_clip.png'
    #     # print(img.shape)
    #     gdal_write_tif(imgoutpath, img, 215, 215, 3)
    #     # clrpath = 'D:\\毕业\\博士论文\\毕业论文\\图\\第二章\\年度数据集图像\\labels_with_year.clr'
    #     # draw_img_with_arcgisclrlist(imgpath, imgoutpath, clrpath)

    import os
    for date in ['10_17', '9_12', '8_04', '7_14', '6_29', '5_20']:
        for net in ['FCN', 'SegNet', 'Base']:
            imgpath = 'D:\\Code\\LULC\\XZY_DeepLearning_Framework\\result\\Spatial_paper\\Segment\\'+date+'\\allchannel_train\\'+net+'\\pred_seg_pre.png'
            imgoutpath = 'C:\\Users\\25321\\Desktop\\新建文件夹\\'+date+'_'+net+'.png'
            clrpath = 'D:\\毕业\\博士论文\\毕业论文\\图\\第二章\\月度数据集图像\\'+date+'_colormap.clr'
            draw_img_with_arcgisclrlist(imgpath, imgoutpath, clrpath)