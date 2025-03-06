import matplotlib.pyplot as plt
import numpy as np
import random
if __name__ == "__main__":
    def generate_unique_colors(num_colors):
        colors = set()

        while len(colors) < num_colors:
            # 生成随机颜色
            color = (random.random(), random.random(), random.random())  # RGB值范围在[0, 1]
            colors.add(color)

        return list(colors)
    unique_colors = [
        [88/255,97/255,172/255],
        [25/255,153/255,178/255],
        [232/255,68/255,69/255]
    ]
    
    ts = [2,3,5]
    values = []
    root_path = "/home/chenyi/gaussian-opacity-fields-main/gaussian-opacity-fields-main/temp/planeocc"
    for t in ts:
        txt_path = root_path + f"/{t}.txt"
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            numbers = np.asarray([float(line.strip()) for i,line in enumerate(lines) if i % 10 == 0])
            values.append(numbers)
    # values = [678,775,874]
    values = [1014.7893,931.2135,875.9146]
    
    plt.figure(figsize=(10, 6))
    
    # plt.ylim(0, 1) 
    # plt.xlim(0, 30000)
    
    plt.ylim(0, 1200) 
    # plt.xlim(0, 30000)
    
    # x = np.linspace(0,30000,30)
    x = ["2","3","5"]
    plt.xlabel('τ',fontsize=16, fontweight='bold', color='darkblue', labelpad=10)
    plt.ylabel('Number of Gaussians/w',fontsize=16, fontweight='bold', color='darkblue', labelpad=10)
    
    # plt.title('The change of G_2d proportion during training',fontsize=15, fontweight='bold', color='black',loc='center', pad=20,fontfamily='serif',fontname='Times New Roman')
    # marker='o', linestyle='-', color='b', linewidth=2, markersize=8, label='线 1'
    plt.title('Number of Gaussians after training completion',fontsize=15, fontweight='bold', color='black',loc='center', pad=20,fontfamily='serif',fontname='Times New Roman')
    
    t = [2,3,5]
    linestyles = ["-","--",":"]
    markers = ["o","s","^"]
    # plt.axvspan(0, 3000, facecolor='gray', alpha=0.5)

    # 显示图表
    # plt.tight_layout()  # 自动调整布局
    # for i in range(3):
    #     # 绘制多个折线
    #     # plt.plot(x, values[i], color=unique_colors[i], label=f'τ={t[i]}',markersize=8,linewidth=3,linestyle=linestyles[i],marker = markers[i])
    #     plt.bar(x, values[i], color=unique_colors[i],edgecolor = "black",label=f'τ={t[i]}')
    plt.bar(x, values, color=unique_colors,label = [f'τ={t[i]}' for i in range(3)],width = 0.5)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加图例
    plt.legend(fontsize=14, 
           loc='upper center', 
           shadow=True, 
           fancybox=True, 
           framealpha=0.8, 
        #    edgecolor='black',
           ncol=3)
    
    # plt.axvline(x=3000, color='gray', linestyle='--', linewidth=2)
    # plt.text(3100, 0.6, 'start planarize', fontsize=13, color='red', ha='left', va='bottom',fontweight='bold')
    plt.savefig('test.png', dpi=600)
    
    

    # from mpl_toolkits.mplot3d import Axes3D

    # # 创建数据
    # x = np.array([1, 2, 3, 4, 5])
    # y = np.array([1, 2, 3, 4, 5])
    # z = np.zeros_like(x)  # 所有柱子的底部高度为0

    # # 设置柱子的高度
    # dx = np.ones_like(x)  # 每个柱子的宽度
    # dy = np.ones_like(y)  # 每个柱子的深度
    # dz = np.random.randint(1, 10, size=x.shape)  # 随机生成每个柱子的高度

    # # 创建三维图
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')

    # # 绘制三维柱状图
    # ax.bar3d(x, y, z, dx, dy, dz, color='cyan', alpha=0.7)

    # # 设置坐标轴标签
    # ax.set_xlabel('x')
    # ax.set_ylabel('z')
    # ax.set_zlabel('y')

    # # 设置标题
    # # ax.set_title('')
    
    # plt.savefig('test.png', dpi=600)
        
