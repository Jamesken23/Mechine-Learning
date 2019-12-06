import matplotlib.pyplot as plt
import numpy as np


def paint_curve(x, y):
    """
    绘制连续的曲线图
    :param x: x轴数据
    :param y: y轴数据
    :return: None
    """
    # 使用plt.plot()绘图。"lw"-线条的宽度；"ls"-线条的风格样式："-", "--", '-.', ':'
    # label-曲线对应的标签，一般在四个边角出现；color-曲线对应的颜色：'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'
    # marker-标记风格，'o', '*, '+'
    plt.plot(x, y, lw=2, ls=':', label="xinge", color="r", marker="*")
    # 设置图案位置
    plt.legend()

    # 调整坐标轴的数值显示范围
    plt.xlim(xmin=-2, xmax=12)
    plt.ylim(ymin=-2, ymax=2)

    # 设置坐标轴的名称.fontsize-字体大小；horizontalalignment-坐标轴名称排列的方向：center’, ‘right’, ‘left’
    plt.xlabel("x-axis", fontsize=16, horizontalalignment="center")
    plt.ylabel("y-axis", fontsize=16, horizontalalignment="center")

    # 调整网格线
    plt.grid(ls=":", color="b")

    # 设置水平参考线和垂直参考线
    plt.axhline(y=0, lw=2, ls="--", color="g")
    plt.axvline(x=0, lw=2, ls="--", color="g")

    # 设置平行于x轴/y轴的参考区域
    # facecolor-设置区域颜色；alpha-设置透明度
    plt.axhspan(ymin=-0.25, ymax=0.25, facecolor="purple", alpha=0.3)
    plt.axvspan(xmin=3, xmax=6, facecolor="g", alpha=0.3)

    # 使用annotate()对曲线添加注释
    # xy: 指示点的坐标，即我们希望注释箭头指向的点的坐标；
    # xytext: 注释文本左端的坐标(不是文本中心的坐标)
    # weight: 注释文本的字体粗细风格，'light', 'normal', 'regular', 'book', 'medium', 'roman'
    # color: 注释文本的颜色
    # arrowstyle: 箭头类型，'->', '|-|', '-|>'
    # connectionstyle: 连接类型，'arc3', 'arc', 'angle', 'angle3'
    plt.annotate('maximum',
                 xy=(np.pi * 3 / 2, -1),
                 xytext=(np.pi * 3 / 2 - 0.6, -0.7),
                 weight='light',
                 color='r',
                 arrowprops={
                     'arrowstyle': '->',
                     'connectionstyle': 'arc3',
                     'color': 'r',
                     'alpha': 0.3
                 })

    # 使用text生成绝对位置的注释
    plt.text(np.pi * 3 / 2 - 0.7, -1.7, s='minimum', weight='regular', color='r', fontsize=12)
    plt.text(np.pi * 3 / 2 - 1, 1.7, s='y=sin(x)', weight='regular', color='r', fontsize=16)

    # 设置图形标题
    plt.title("A Sine Curve")

    plt.show()


def paint_scatter(x, y):
    """
    绘制散点图
    :param x: x轴数据
    :param y: y轴数据
    :return: None
    """
    # 使用plt.plot()绘图。"lw"-线条的宽度；"ls"-线条的风格样式："-", "--", '-.', ':'
    # label-曲线对应的标签，一般在四个边角出现；color-曲线对应的颜色：'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'
    plt.scatter(x, y, lw=2, ls=':', label="xinge", color="r")
    # 设置图案位置
    plt.legend()

    plt.show()


def paint_hist(a, b):
    """
    绘制直方图
    :param x: x轴数据
    :param y: y轴数据
    :return: None
    """
    x = np.random.randint(0, 100, 100)  # 生成【0-100】之间的100个数据,即 数据集
    bins = np.arange(0, 101, 10)  # 设置连续的边界值，即直方图的分布区间[0,10],[10,20]...
    width = 10  # 柱状图的宽度
    # 生成直方图
    # density: bool，默认为false，显示的是频数统计结果，为True则显示频率统计结果
    # histtype: 可选{'bar', 'barstacked', 'step', 'stepfilled'}之一，默认为bar，推荐使用默认配置，step使用的是梯状，
    frequency_each, _, _ = plt.hist(x, bins, color='deepskyblue', width=width, alpha=0.7)
    plt.xlabel('scores')
    plt.ylabel('count')
    plt.xlim(0, 100)  # 设置x轴分布范围
    # 利用返回值来绘制区间中点连线
    plt.plot(bins[1:] - (width // 2), frequency_each, color='palevioletred')
    plt.show()


def paint_multi_graph():
    """
    绘制多张子图
    :return:
    """
    x = np.linspace(0, 5)
    y1 = np.sin(np.pi * x)
    y2 = np.sin(np.pi * x * 2)

    # 直接用plt.figure()显示不了子图
    plt.figure(num="xinge", figsize=(10, 10))

    # subplot()函数返回两个对象。一个是fig画布，一个是ax子图：ax.plot()
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(x, y1, ls="-", label="sin(pi*x)", c="b")
    # 子图设置x,y轴标签时写法不一样。set_xlabel()
    ax[0].set_xlabel("x1 label")
    ax[0].set_ylabel("y1 value")
    ax[0].legend()

    ax[1].plot(x, y2, ls="-", label="sin(pi*x*2)", c="r")
    ax[1].set_xlabel("x1 label")
    ax[1].set_ylabel("y2 value")
    ax[1].legend()

    plt.show()


def paint_figure():
    """
    使用画布来做图
    :return:
    """
    # 添加画布。num-画布名称：int,string
    fig = plt.figure(num="xinge", figsize=(8, 8), facecolor="r")
    # 使用画布分别添加子图
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.random.rand(11).cumsum()
    y = np.random.rand(11).cumsum()
    ax1.plot(x, y, 'c*-', label='ax1', linewidth=2)

    ax2.plot(x, y, 'm.-.', label='ax2', linewidth=1)
    ax1.legend()
    ax1.set_title('hahaha')
    ax2.legend()
    ax2.set_title('xixixi')
    ax1.set_ylabel('hengzhou')
    ax2.set_ylabel('zongzhou')

    plt.show()


if __name__ == "__main__":
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    y_sca = np.random.rand(100)
    # paint_curve(x, y)
    # paint_scatter(x, y_sca)

    # np.random.seed(0)  # 设置一个随机种子0
    # mu, sigma = 100, 20  # 均值为100，方差为20
    # a = np.random.normal(mu, sigma, size=100)
    # b = 20
    # paint_hist(a, b)

    # paint_multi_graph()
    paint_figure()
