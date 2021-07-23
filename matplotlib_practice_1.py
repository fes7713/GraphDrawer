# import matplotlib.pyplot as plt
#
#
# def on_pressed(event):
#     plot_objects.set_data(x, y**2)
#     fig.canvas.draw()
#
# # 対象データ
# x = [1, 2, 3, 4, 5]  # x軸の値
# y = [100, 300, 200, 500, 0]  # y軸の値
#
# # figureを生成する
# fig, ax = plt.subplots()
#
# # axesにplot
# plot_objects = ax.plot(x, y)
# # 表示する
# fig.canvas.mpl_connect("button_press_event", on_pressed)
# # fig.canvas.mpl_connect("motion_notify_event", pthandler.on_motion)
# # fig.canvas.mpl_connect("pick_event", pthandler.on_picked)
# plt.show()
#
#
#

import numpy as np
import matplotlib.pyplot as plt

def on_pressed(event):
    # x = line.get_xdata()  # x
    y = line.get_ydata()  # y
    print(x.size)
    print(y.size)
    if (y == y1).all():
        line.set_data(x, y2)
    else:
        line.set_data(x, y1)
    plt.draw()

x  = np.linspace(-np.pi, np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
print(y1==y2)
line, = plt.plot(x, y1)
plt.connect("button_press_event", on_pressed)
plt.show()

