# import numpy as np
# import matplotlib.pyplot as plt
# # https://matplotlib.org/tutorials/introductory/pyplot.html より
# def f(t):
#     return np.exp(-t) * np.cos(2*np.pi*t)
#
# t1 = np.arange(0.0, 5.0, 0.1)
# t2 = np.arange(0.0, 5.0, 0.02)
#
# plt.figure(1)
# plt.subplot(211)
# plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
#
# plt.subplot(212)
# plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
# plt.show()
#
# x = np.linspace(0, 2*np.pi, 100)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x, np.sin(x), label='sin(x)')
# ax1 = ax.twinx()
# ax1.plot(x, 2*np.cos(x), c='C1', label='2*cos(x)')
#
# ax.legend()
# ax1.legend()


import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame({"GameID":[1,2,3,4,5],
                     "Date":["2018-03-30", "2018-03-31", "2018-04-01", "2018-04-03", "2018-04-04"],
                     "H":[4,7,9,6, 14],
                     "HR":[1,0,1,0,0],
                     "K":[10,8,3,9,5],
                     "BB":[2,5,2,1,4],
                     "R":[2,2,1,2,5]})
df = data.iloc[0:10]
print(df.head())
fig, ax = plt.subplots()

ax = fig.add_subplot(111, xlabel=df.index.name, ylabel='number')

ax.plot(df['H'])
ax.plot(df['HR'], 'rs:', label='HR', ms=10, mew=5, mec='green')
ax.plot(df['K'], marker='^', linestyle='-.')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, xlabel=df['H'].name, ylabel=df['R'].name)

ax.scatter(df['H'], df['R'], c='blue')

plt.show()

x = df['H']
y = df['R']
cm = df['HR'].astype(str)

fig = plt.figure()
ax = fig.add_subplot(111, xlabel=x.name, ylabel=y.name,
                     xlim=(0, 15), ylim=(0, 10))

ax.scatter(x, y, s=x**2, c='C'+cm, zorder=10)
ax.grid(c='gainsboro', zorder=9)
plt.show()