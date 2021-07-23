import numpy as np
import pandas as pd
import collections

# def spline_interpolation(x, y, precision):
#     df_coordinates = pd.DataFrame({"x":x, "y":y})
#
#     df_coordinates = df_coordinates.drop_duplicates(subset='x')
#
#     x = df_coordinates["x"]
#     y = df_coordinates["y"]
#
#     x_1 = np.roll(x, -1)
#     x_1[-1] = np.NaN
#
#     x_2 = np.roll(x_1, -1)
#     x_2[-1] = np.NaN
#
#     h = x_1 - x
#     h_1 = x_2 - x_1
#     h__1 = np.roll(h, 1)
#     h__1[0] = np.NaN
#
#     y_1 = np.roll(y, -1)
#     y__1 = np.roll(y, 1)
#     y__1[0] = np.NaN
#
#     v = 6*((y_1-y)/h-(y-y__1)/h__1)
#     v = v[1:-1]
#
#     h_mat = np.diag(h)
#     h_1_mat = np.diag(h_1)
#     h_1_up_mat = np.roll(h_mat, -1, axis=0)
#     h_1_up_mat[-1] = 0
#     h_1_left_mat = np.roll(h_mat, -1, axis=1)
#     h_1_left_mat[:, -1] = 0
#     hh_1_mat = 2*(h_mat+h_1_mat)
#
#     fMat = (hh_1_mat+h_1_left_mat+h_1_up_mat)[:-2,:-2]
#     fMat_inv = np.linalg.inv(fMat)
#
#     u_1 = np.dot(fMat_inv, v)
#     u_1 = np.hstack((u_1, np.array([0, 0])))
#     u = np.roll(u_1, 1)
#     u_1[-1] = np.NaN
#
#     a = (u_1 - u) / (6 * (x_1 - x))
#     b = u/2
#     c = ((y_1 - y) / h) - (1 / 6) * h * (2 * u + u_1)
#     d = y
#
#     sample_x = pd.Series(np.linspace(np.min(x), np.max(x), precision))
#
#     data = {"x":x, "y":y, "a":a, "b":b, "c":c, "d": d, "u":u}
#     df = pd.DataFrame(data)
#     cut_data = pd.cut(sample_x, df["x"], labels=df["x"][0:-1])
#     result = []
#
#     for i in range(sample_x.size):
#         data = np.ravel(df[df["x"]==cut_data[i]]).tolist()
#         data = [sample_x[i]] + data
#         result.append(data)
#
#     df_final = pd.DataFrame(result, columns=["sample_x", "x", "y", "a", "b", "c", "d", "u"])
#
#     y_height = df_final["a"]*(df_final["sample_x"]-df_final["x"])**3 + df_final["b"]*(df_final["sample_x"]-df_final["x"])**2 + df_final["c"]*(df_final["sample_x"]-df_final["x"])+df_final["d"]
#
#     return df_final["sample_x"], y_height


def spline_interpolation(x, y, precision):
    if type(x) is not np.ndarray:
        x = np.array(x).astype(float)
    if type(y) is not np.ndarray:
        y = np.array(y).astype(float)
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

