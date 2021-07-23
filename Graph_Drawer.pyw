import matplotlib.pyplot as plt
import numpy as np
import collections

LEFT_CLICK = 1
RIGHT_CLICK = 3


def spline_interpolation(x, y, precision):
    a = np.unique(x, return_index=True)

    x = a[0]
    y = y[a[1]]

    x_1 = np.roll(x, -1)
    x_1[-1] = np.NaN

    x_2 = np.roll(x_1, -1)
    x_2[-1] = np.NaN

    h = x_1 - x
    h_1 = x_2 - x_1
    h__1 = np.roll(h, 1)
    h__1[0] = np.NaN

    y_1 = np.roll(y, -1)
    y__1 = np.roll(y, 1)
    y__1[0] = np.NaN

    v = 6 * ((y_1 - y) / h - (y - y__1) / h__1)
    v = v[1:-1]

    h_mat = np.diag(h)
    h_1_mat = np.diag(h_1)
    h_1_up_mat = np.roll(h_mat, -1, axis=0)
    h_1_up_mat[-1] = 0
    h_1_left_mat = np.roll(h_mat, -1, axis=1)
    h_1_left_mat[:, -1] = 0
    hh_1_mat = 2 * (h_mat + h_1_mat)

    fMat = (hh_1_mat + h_1_left_mat + h_1_up_mat)[:-2, :-2]
    fMat_inv = np.linalg.inv(fMat)

    u_1 = np.dot(fMat_inv, v)
    u_1 = np.hstack((u_1, np.array([0, 0])))
    u = np.roll(u_1, 1)
    u_1[-1] = np.NaN

    a = (u_1 - u) / (6 * (x_1 - x))
    b = u / 2
    c = ((y_1 - y) / h) - (1 / 6) * h * (2 * u + u_1)
    d = y

    sample_x = np.linspace(np.min(x), np.max(x), precision)
    data = np.array([x, y, a, b, c, d, u])
    bins = data[0]

    index = np.digitize(sample_x, bins=bins[0:-1]) - 1
    index_counter = collections.Counter(index)
    index_counter = list(dict(sorted(index_counter.items())).values())
    result = np.zeros((1 + np.size(data, 0), np.size(sample_x)), float)
    result[0, :] = sample_x

    for i in range(len(index_counter)):
        sample_tile = np.tile(data[:, i], index_counter[i])
        sample_tile_reshape = sample_tile.reshape(index_counter[i], np.size(data, 0))
        sample_tile_T = sample_tile_reshape.T

        result[1:, sum(index_counter[0:i]):sum(index_counter[0:i + 1])] = sample_tile_T

    data_dict = {"sample_x": result[0], "x": result[1], "y": result[2], "a": result[3], "b": result[4],
                 "c": result[5], "d": result[6], "u": result[7]}

    y_height = data_dict["a"] * (data_dict["sample_x"] - data_dict["x"]) ** 3 + data_dict["b"] * \
               (data_dict["sample_x"] - data_dict["x"]) ** 2 + data_dict["c"] * \
               (data_dict["sample_x"] - data_dict["x"]) + data_dict["d"]

    return data_dict["sample_x"], y_height


def update(event_handler):
    """post processing updating artist objects"""

    def event_handler_decorated(self, event, *args, **kwargs):
        event_handler(self, event, *args, **kwargs)

        sorted_index = np.argsort(self.x_data)
        self.x_data = self.x_data[sorted_index]
        self.y_data = self.y_data[sorted_index]
        self.select_index = sorted_index[self.select_index]

        self.x_interpolation, self.y_interpolation = spline_interpolation(self.x_data, self.y_data,
                                                                          20 * self.x_data.size)
        self.line.set_data(self.x_interpolation, self.y_interpolation)
        self.points.set_data(self.x_data, self.y_data)
        self.cursor.set_data(event.xdata, event.ydata)

        plt.draw()

    return event_handler_decorated


class GraphDrawer:

    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        # coords
        self.x_data = np.array([0, 8, 9, 27], dtype=float)
        self.y_data = np.array([9, 10, 11, 14], dtype=float)

        self.x_interpolation, self.y_interpolation = spline_interpolation(self.x_data, self.y_data,
                                                                          20 * self.x_data.size)
        self.line, = plt.plot(self.x_interpolation, self.y_interpolation)
        self.points, = self.ax.plot(self.x_data, self.y_data, "o", mew=2, mec="g")
        self.points.set_pickradius(10)
        self.points.set_picker(True)
        self.cursor, = ax.plot(self.x_data[0], self.y_data[0], "o", mew=2, mec="r")

        self.is_picking_object = False
        self.select_index = None
        plt.title("Right Click to Add a Point and Left Click to Remove a Point")
    def on_picked(self, event):
        self.select_index = event.ind[0]
        if event.mouseevent.button == LEFT_CLICK:
            self.is_picking_object = True
        if event.mouseevent.button == RIGHT_CLICK:
            self.remove_point()

    @update
    def on_pressed(self, event):
        if self.is_picking_object:
            return
        if event.button == LEFT_CLICK:
            print("Left")
            self.add_point(event.xdata, event.ydata)
        if event.button == RIGHT_CLICK:
            print("Right")

    @update
    def on_release(self, event):
        if self.is_picking_object:
            self.move_point(event.xdata, event.ydata)

        self.is_picking_object = False

    @update
    def on_motion(self, event):
        if not self.is_picking_object:
            return
        self.move_point(event.xdata, event.ydata)

    def add_point(self, x, y):
        self.x_data = np.append(self.x_data, x)
        self.y_data = np.append(self.y_data, y)

    def remove_point(self):
        self.x_data = np.delete(self.x_data, self.select_index)
        self.y_data = np.delete(self.y_data, self.select_index)

    def move_point(self, x, y):
        self.x_data[self.select_index] = x
        self.y_data[self.select_index] = y


def main():
    fig, ax = plt.subplots()
    drawer = GraphDrawer(fig, ax)
    plt.grid(True)

    fig.canvas.mpl_connect('pick_event', drawer.on_picked)
    fig.canvas.mpl_connect("button_press_event", drawer.on_pressed)
    fig.canvas.mpl_connect("motion_notify_event", drawer.on_motion)
    fig.canvas.mpl_connect("button_release_event", drawer.on_release)
    plt.show()


if __name__ == '__main__':
    main()
