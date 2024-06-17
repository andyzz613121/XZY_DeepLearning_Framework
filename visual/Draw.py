from turtle import width
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 
from matplotlib import ticker, cm
import numpy as np
import matplotlib as mpl
from pylab import xticks,yticks
from matplotlib import colors
import random
def draw_heatmap1(img, save_path, norm=False):
    fig, ax = plt.subplots()
    # position = fig.add_axes([0.125, 0.19, 0.2, 0.05])
    # img = ax.imshow(img, cmap=cm.RdYlBu_r)
    # norm = colors.Normalize(vmin=-1., vmax=1.)
    
    cbar = plt.pcolor(img, cmap=cm.RdYlBu_r, vmin=-1., vmax=1.)
    
    # cbar.ax.tick_params(labelsize=6)
    # tick_locator = ticker.MaxNLocator(nbins=3)
    # cbar.locator = tick_locator
    # cbar.update_ticks()
    # ax.clim(0, 2) 
    ax.set_aspect(1)
    fig.savefig(save_path, dpi=600)

def draw_heatmap(img, save_path, norm=False):
    '''
        Input: img[H*W]
        ColorBar can be found in: https://www.jb51.net/article/258753.htm
        Out: color_img[]
    '''
    if norm == True:
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        mins, maxs = np.min(img), np.max(img)
    else:
        mins, maxs = 0, 1.5
    print(np.min(img), np.max(img))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # cax = ax.matshow(img, interpolation='nearest', cmap=cm.RdYlBu_r)
    # fig.colorbar(cax)

    # ax.spines['bottom'].set_position(('data', 0))
    # ax.xaxis.set_ticks_position(('top'))
    # im = ax.imshow(img, vmin=mins, vmax=maxs)
    
    # fig.subplots_adjust(right=0.8)

    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    
    X,Y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    cs = ax.contourf(X, Y, img, 200, cmap=cm.RdYlBu_r, vmin=mins, vmax=maxs) # RdYlBu_r是RdYlBu色带的倒转， 100是分级数
    cbar = fig.colorbar(cs)
    # fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # cbar.set_clim(mins, maxs)    

    ax.invert_yaxis()
    # ax.xaxis.set_ticks_position('top')
    
    ax.set_aspect(1)
    plt.xlabel(u'Columns (Pixel)', fontsize=18)
    plt.ylabel(u'Rows (Pixel)', fontsize=18)
    mpl.pyplot.axis('off')
    

    fig.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.clf()
    plt.close()

def draw_curve(x, y, save_path, xticks=None, yticks=None, x_ticknum=5, y_ticknum=5, x_label='Epoch', y_label='Validation Loss'):
    plt.figure(figsize=(6, 6))

    # 画第1条折线，参数看名字就懂，还可以自定义数据点样式等等。
    plt.plot(x, y, color='#FF0000', linewidth=2.0)

    # plt.axes().get_xaxis().set_visible(False) # 隐藏x坐标轴
    # plt.axes().get_yaxis().set_visible(False) # 隐藏y坐标轴
    # # 给折线上的数据点加上数值，前两个参数是坐标，第三个是数值，ha和va分别是水平和垂直位置（数据点相对数值）。
    # for a, b in zip(x, y):
    #     plt.text(a, b, '%d'%b, ha='center', va= 'bottom', fontsize=18)

    # # 画水平横线，参数分别表示在y=3，x=0~len(x)-1处画直线。
    # plt.hlines(3, 0, len(x)-1, colors = "#000000", linestyles = "dashed")

    # 3、添加x轴和y轴刻度标签
    xticks = [i*(max(x)-min(x))/x_ticknum for i in range(x_ticknum+1)] if xticks is None else xticks
    yticks = [round(i*(max(y)-min(y))/y_ticknum, 3) for i in range(y_ticknum+1)] if yticks is None else yticks
    
    plt.xticks(xticks, fontsize=12, rotation=0)
    plt.yticks(yticks, fontsize=12)
    # # 添加x轴和y轴标签
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    # 4、绘制折线图标题和图例
    # 标题
    # plt.title(u'Title', fontsize=18)
    # 图例
    # plt.legend(fontsize=18)

    # 5、保存完成
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    # 显示图片
    # plt.show()
    plt.close()

def draw_bar(x, y, save_path, xticks=None, yticks=None, x_ticknum=5, y_ticknum=5, x_label='Epoch', y_label='Validation Loss'):
        
    plt.figure(figsize=(10, 6))
    #绘制纵向柱状图
    # plt.bar(x,y,align = "center", color=["r","g","b"], tick_label = xticks, hatch = "/", ec = 'gray')
    plt.bar(x,y,align = "center", color='#0000FF')
    #hatch定义柱图的斜纹填充，省略该参数表示默认不填充; ec边框颜色为灰色。
    
    #绘制X、Y轴标签
    # plt.xlabel(u"样品编号") #u代表对字符串进行unicode编码。对中文表明所需编码，防止出现乱码。
    # plt.ylabel(u"库存数量")
    plt.xticks(xticks, fontsize=12, rotation=0)
    plt.yticks(fontsize=12)
    #绘制柱状图标题
    # plt.title("带颜色的柱状图")


    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=600)

    plt.close()

def draw_bar_2y(y1_list, y2_list, save_path, xticks=None, y1_ticks=None, y2_ticks=None, x_ticknum=5, y_ticknum=5):
    '''
        Input:
            y1_list: 第一组y轴的numpy list -> [np.ary1, np.ary2]
            y2_list: 第一组y轴的numpy list -> [np.ary1, np.ary2]
    '''
    # 生成一些示例数据
    # x: x轴的numpy数组
    x = np.arange(len(y1_list[0]))
        
    # 创建一个图形和两个y轴  
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10, 10)
    ax2 = ax1.twinx()
    import random
    #绘制折线图
    for i in range(len(y1_list)):
        y1 = y1_list[i]
        red = random.randint(0, 255)/255
        green = random.randint(0, 255)/255
        blue = random.randint(0, 255)/255
        color = (red, green, blue)
        print(color)
        color = (0.3137254901960784, 0.09803921568627451, 0.8901960784313725)
        line1 = ax1.plot(x, y1,label='精度值', color=color, marker='o', ls='-.')
    color_list = [(0.9568627450980393, 0.21568627450980393, 0.16470588235294117),(0.9411764705882353, 0.6862745098039216, 0.5803921568627451),
                  (0.9568627450980393, 0.21568627450980393, 0.16470588235294117),(0.9411764705882353, 0.6862745098039216, 0.5803921568627451)]
    for i in range(len(y2_list)):
        y2 = y2_list[i]
        red = random.randint(0, 255)/255
        green = random.randint(0, 255)/255
        blue = random.randint(0, 255)/255
        color = (red, green, blue)
        print(color)
        color = color_list[i]
        line2 = ax2.plot(x, y2, label='类别数', color=color, marker=None, ls='--')  
    
    # 设置x轴和y轴的标签，指明坐标含义  
    ax1.set_xlabel('x轴', fontdict={'size': 16})  
    ax1.set_ylabel('y1轴',fontdict={'size': 16})  
    ax2.set_ylabel('y2轴',fontdict={'size': 16})
    if y1_ticks is not None:
        ax1.set_yticks(y1_ticks)
    if y2_ticks is not None:
        ax2.set_yticks(y2_ticks)
    #添加图表题  
    plt.title('双y轴折线图')  
    #添加图例  
    # plt.legend()  
    # 设置中文显示  
    plt.rcParams['font.sans-serif']=['SimHei']  
    #展示图片 
    plt.savefig(save_path) 
    plt.show()

def draw_lidar(x_list, y_list, save_path, xticks=None, yticks=None):
    '''
        Input:
            x_list: [[1...n], [1...n]...] 不同的系列
            y_list: [1...n] 标签的列表
    '''
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "FangSong" # 设置字体
    mpl.rcParams["axes.unicode_minus"]=False # 正常显示负号
    # mpl.rcParams["font.family"] = "Times New Roman" # 设置字体
    angles = np.linspace(0,2*np.pi,len(y_list),endpoint=False)
    angles=np.concatenate((angles,[angles[0]]))
    labels = y_list+[y_list[0]] # 每个系列对应的标签

    fig = plt.figure(figsize=(10,10))       #facecolor 设置框体的颜色
    ax = plt.subplot(111, polar=True)     #将图分成1行1列，画出位置1的图；设置图形为极坐标图
    color_list = [(0.9568627450980393, 0.21568627450980393, 0.16470588235294117),(0.9411764705882353, 0.6862745098039216, 0.5803921568627451),
                (0.043137254901960784, 0.32941176470588235, 0.9607843137254902),(0.12549019607843137, 0.5176470588235295, 0.24313725490196078)]
    for i in range(len(x_list)):
        red = random.randint(0, 255)/255
        green = random.randint(0, 255)/255
        blue = random.randint(0, 255)/255
        color = (red, green, blue)
        if i < len(color_list):
            color = color_list[i]
        x_list[i]=np.concatenate((x_list[i],[x_list[i][0]]))
        
        ax.plot(angles, x_list[i],'bo-',color=color,linewidth=1)
        plt.thetagrids(angles*180/np.pi,labels)
    # ax.grid(True)
    # ax.set_yticklabels(yticks)
    # ax.set_yticklabels([])
    # plt.show()
    plt.savefig(save_path, dpi=600)

def draw_multiseries(x, y, save_path, xticks=None, yticks=None, x_ticknum=5, y_ticknum=5, x_label='Epoch', y_label='Validation Loss', type='line'):
    '''
        Input:
            type: line or bar   折线图或柱状图
    '''
        
    # 创建一个图形和两个y轴  
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    
    #绘制折线图
    color_list = [(0.9568627450980393, 0.21568627450980393, 0.16470588235294117),(0.043137254901960784, 0.32941176470588235, 0.9607843137254902),
                  (0.9411764705882353, 0.6862745098039216, 0.5803921568627451),(0.12549019607843137, 0.5176470588235295, 0.24313725490196078)]

    for i in range(len(x)):
        red = random.randint(0, 255)/255
        green = random.randint(0, 255)/255
        blue = random.randint(0, 255)/255
        color = (red, green, blue)
        print(color)
        if i < len(color_list):
            color = color_list[i]
        if type == 'line':
            ax.plot(y[0],x[i],label=str(i), color=color, marker='o', ls='--')
        if type == 'bar':
            ax.bar(y[0],x[i],label=str(i), color=color)
    
    # # 设置x轴和y轴的标签，指明坐标含义  
    # ax.set_xlabel('x轴', fontdict={'size': 16})  
    # ax.set_ylabel('y1轴',fontdict={'size': 16})
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    # #添加图表题  
    # plt.title('双y轴折线图')  
    #添加图例  
    # plt.legend()  
    # 设置中文显示  
    plt.rcParams['font.sans-serif']=['SimHei']  
    #展示图片 
    plt.savefig(save_path, dpi=600) 
    # plt.show()

if __name__ == '__main__':
    # import sys
    # base_path = '..\\XZY_DeepLearning_Framework\\'
    # sys.path.append(base_path)
    # from data_processing.excel import *
    # aa = read_excel('C:\\Users\\25321\\Desktop\\第6章.xls')
    # items = get_total_item(aa)
    # path = 'C:\\Users\\25321\\Desktop\\mIoU.jpg'
    # x = np.array(items[:7])
    # print(x)
    # draw_lidar(items[:7], items[7], yticks=[5,6,7,8,9,10,'tiome'], save_path=path)
    x = np.array([[1, 	2, 	3, 	4, 	5],
                   [1, 	2, 	3, 	4, 	5]])
    y = np.array([[0.845, 	0.850, 	0.850, 	0.850, 	0.849],
                                 [0.809, 	0.817, 	0.819, 	0.823, 	0.821]])
    # y = np.array([[0.750, 	0.750, 	0.753, 	0.748 ,	0.747 ],
    #                              [0.738 ,	0.746 ,	0.752 	,0.734 ,	0.729 ]])
    # y = np.array([[0.611 ,	0.624 	,0.625 	,0.627 ,	0.620 ],
    #                              [0.565 ,	0.582 ,	0.582 	,0.585 	,0.569 ]])
    # y = np.array([[0.808 ,	0.813, 	0.814 ,	0.813 ,	0.812 ],
    #                              [0.764 ,	0.773, 	0.775 ,	0.778 ,	0.775 ]])
    draw_multiseries(y,x,yticks=[x/100 for x in range(80, 90, 2)],save_path='C:\\Users\\25321\\Desktop\\OA.jpg')