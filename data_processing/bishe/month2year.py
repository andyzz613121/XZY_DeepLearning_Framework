import numpy as np
'''
    将基于月度的预测结果映射为年度结果
'''
def month2year(month_preimg, month):
    month_maxclass = np.max(month_preimg)
    posidx_list = []
    for i in range(month_maxclass+1):
        posidx_list.append(month_preimg==i)
    
    if month == '5_20':
        month_preimg[posidx_list[1]] = 7
        month_preimg[posidx_list[2]] = 6
        month_preimg[posidx_list[3]] = 8
        month_preimg[posidx_list[4]] = 9
        month_preimg[posidx_list[5]] = 10
        month_preimg[posidx_list[6]] = 11
    elif month == '6_29':
        month_preimg[posidx_list[2]] = 7
        month_preimg[posidx_list[4]] = 8
        month_preimg[posidx_list[5]] = 9
        month_preimg[posidx_list[6]] = 10
        month_preimg[posidx_list[7]] = 11
    elif month == '7_14':
        month_preimg[posidx_list[4]] = 5
        month_preimg[posidx_list[5]] = 7
        month_preimg[posidx_list[6]] = 8
        month_preimg[posidx_list[7]] = 9
        month_preimg[posidx_list[8]] = 10
        month_preimg[posidx_list[9]] = 11
    elif month == '8_04':
        month_preimg[posidx_list[6]] = 7
        month_preimg[posidx_list[7]] = 8
        month_preimg[posidx_list[8]] = 9
        month_preimg[posidx_list[9]] = 10
        month_preimg[posidx_list[10]] = 11
    elif month == '9_12':
        month_preimg[posidx_list[1]] = 2
        month_preimg[posidx_list[2]] = 3
        month_preimg[posidx_list[3]] = 4
        month_preimg[posidx_list[4]] = 5
        month_preimg[posidx_list[5]] = 7
        month_preimg[posidx_list[6]] = 8
        month_preimg[posidx_list[7]] = 9
        month_preimg[posidx_list[8]] = 10
        month_preimg[posidx_list[9]] = 11
    elif month == '10_17':
        month_preimg[posidx_list[1]] = 7
        month_preimg[posidx_list[2]] = 8
        month_preimg[posidx_list[3]] = 9
        month_preimg[posidx_list[4]] = 10
        month_preimg[posidx_list[5]] = 11
    return month_preimg