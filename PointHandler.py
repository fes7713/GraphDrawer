"""
https://matplotlib.org/users/event_handling.html
"""

from matplotlib import pyplot as plt
import numpy as np

LEFT_CLICK = 1
RIGHT_CLICK = 3


def update(event_handler):
    """post processing updating artist objects"""

    def event_handler_decorated(self, *args, **kwargs):
        event_handler(self, *args, **kwargs)
        self.plot_objects.set_data(self.xs, self.ys)
        self.fig.canvas.draw()

    return event_handler_decorated




class PointHandler:

    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        # coords
        self.xs = np.array([])
        self.ys = np.array([])
        # artists
        self.plot_objects, = ax.plot(
            self.xs, self.ys, 'bo', picker=5, mew=2, mec='g')

        plt.grid(True)


    @update
    def on_pressed(self, event):
        if event.button != LEFT_CLICK:
            return
        if event.inaxes != self.ax:
            return
        self.add_point(event.xdata, event.ydata)


    @update
    def on_motion(self, event):
        test = 0

    @update
    def on_picked(self, event):
        test = 0

    @update
    def on_release(self, event):
        test = 0

    def add_point(self, x, y):
        self.xs = np.append(self.xs, x)
        self.ys = np.append(self.ys, y)

    def move_point(self, x, y):
        self.xs[self.select_index] = x
        self.ys[self.select_index] = y

    def remove_point(self):
        self.xs = np.delete(self.xs, self.select_index)
        self.ys = np.delete(self.ys, self.select_index)


def main():
    fig, ax = plt.subplots()
    ax.set_title(
        "Left click to build point. Right click to remove point.")
    pthandler = PointHandler(fig, ax)
    # regist event handler
    # the order of mpl_connect is important
    fig.canvas.mpl_connect("button_press_event", pthandler.on_pressed)
    fig.canvas.mpl_connect("motion_notify_event", pthandler.on_motion)
    fig.canvas.mpl_connect("pick_event", pthandler.on_picked)
    fig.canvas.mpl_connect("button_release_event", pthandler.on_release)
    plt.show()


if __name__ == '__main__':
    main()