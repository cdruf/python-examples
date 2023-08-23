import math
import os

import pandas as pd
from IPython import display
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from inventory_management.economic_order_quantity import EOQ

eoqs = [EOQ(D=12000, A=a, v=1.0, r=0.1) for a in range(20, 101)]
data = []
for eoq in eoqs:
    eoq.compute()
    data.append((eoq.A, eoq.Q, eoq.T))

df = pd.DataFrame(data, columns=['A', 'EOQ', 'EOI'])
print(df.head())

eoqs[0].plot()


# %%

# matplotlib.use('TkAgg')


def get_xs_and_ys(eoq: EOQ):
    order_cycles_per_year = math.ceil(1.0 / eoq.T)
    xs = []
    ys = []
    days = 365 * eoq.T
    for i in range(order_cycles_per_year):
        xs.append(i * days)
        ys.append(eoq.Q)
        xs.append((i + 1) * days)
        ys.append(0)
    return xs, ys


xs, ys = get_xs_and_ys(eoqs[0])
print(len(xs))
print(len(ys))

# %%

# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(0, 365)
# ax.set_xticks([])
ax.set_ylim(0, df['EOQ'].max())
# ax.set_yticks([])
ln, = ax.plot(xs, ys)


def update(frame):
    print(frame)
    eoq = eoqs[frame]
    xs, ys = get_xs_and_ys(eoq)
    ln.set_data(xs, ys)
    return ln,


# animation = FuncAnimation(fig, update, interval=10)
animation = FuncAnimation(fig, update, frames=range(len(eoqs)), blit=True)
plt.show()

# %%

print(os.getcwd())

print("Save video to disk")
animation.save("data/video.mp4")
print("Video saved")

# %%
video = animation.to_html5_video()
html = display.HTML(video)
display.display(html)

plt.close()
