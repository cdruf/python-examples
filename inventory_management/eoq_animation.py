import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from inventory_management.economic_order_quantity import EOQ

# %%

eoqs = [EOQ(D=12000, A=a, v=1.0, r=0.1) for a in range(20, 101)]
data = []
for eoq in eoqs:
    eoq.compute()
    data.append((eoq.A, eoq.Q, eoq.T))

df = pd.DataFrame(data, columns=['A', 'EOQ', 'EOI'])
print(df.head())

eoqs[0].plot()

# %%

# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(0, 365)
# ax.set_xticks([])
ax.set_ylim(0, df['EOQ'].max())
# ax.set_yticks([])
ln, = ax.plot(*eoqs[0].get_xs_and_ys())


def update(frame):
    print(frame)
    eoq = eoqs[frame]
    xs, ys = eoq.get_xs_and_ys()
    ln.set_data(xs, ys)
    return ln,


# animation = FuncAnimation(fig, update, interval=10)
animation = FuncAnimation(fig, update, frames=range(len(eoqs)), blit=True)
# plt.show()

print("Save video to disk")
writer = PillowWriter()
animation.save("data/eoq.gif", writer=writer)
print("Video saved")
