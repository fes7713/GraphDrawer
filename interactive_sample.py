import matplotlib.pyplot as plt

def motion(event):
    global gco
    if gco is None:
        return
    x = event.xdata
    y = event.ydata
    gco.set_data(x,y)
    plt.draw()

def onpick(event):
    global gco
    gco = event.artist
    plt.title(gco)

def release(event):
    global gco
    gco = None

gco = None
plt.figure()

plt.plot(0,0,"o",picker=15)
plt.plot(1,0,"o",picker=15)

plt.connect('motion_notify_event', motion)
plt.connect('pick_event', onpick)
plt.connect('button_release_event', release)
plt.show()
# import matplotlib.pyplot as plt
#
# def motion(event):
#     if event.dblclick == 1:
#         plt.title("double click")
#
#     elif event.button == 1:
#         plt.title("left click")
#
#     elif event.button == 3:
#         plt.title("right click")
#
#     plt.draw()
#
# plt.figure()
# plt.connect('button_press_event', motion)
# plt.show()