import matplotlib
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.matplotlib_fname()
import matplotlib.rcsetup as rcsetup
import matplotlib.ticker as ticker
import numpy as np

# from brokenaxes import brokenaxes
print(rcsetup.all_backends)

# %matplotlib qt # => display plots in external window
# %matplotlib inline # => display plots inline

# %%

"""
# Figure: The whole figure: Axes, titles, legends, etc, and the canvas.
"""
fig = plt.figure()  # an empty figure with no axes
fig, ax_lst = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes

# %%

plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()

# %%

# die 1. Liste definiert die x-Koordinaten 
# die 2. Liste definiert die y-Koordinaten
# stückweise linear durch die Punkte
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.show()

# %%

# verschiedene Styles
t = np.arange(0., 5., 0.2)
plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')
plt.show()

# Data frames
data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100
plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()

# Kategoriale Variablen
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]
plt.figure(1, figsize=(9, 3))
plt.subplot(131)  # subplot(xyz) = subplot(x, y, z), numrows, numcols, plot_number 
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()

# Line properties
# ...

'''
Multiple figures and axes
'''
np.random.seed(19680801)
dt = 0.001
t = np.arange(0.0, 10.0, dt)
r = np.exp(-t[:1000] / 0.05)  # impulse response
x = np.random.randn(len(t))
s = np.convolve(x, r)[:len(x)] * dt  # colored noise

# the main axes is subplot(111) by default
plt.plot(t, s)
plt.axis([0, 1, 1.1 * np.min(s), 2 * np.max(s)])
plt.xlabel('time (s)')
plt.ylabel('current (nA)')
plt.title('Gaussian colored noise')

# this is an inset axes over the main axes
a = plt.axes([.65, .6, .2, .2], facecolor='k')
n, bins, patches = plt.hist(s, 400, density=True)
plt.title('Probability')
plt.xticks([])
plt.yticks([])

# this is another inset axes over the main axes
a = plt.axes([0.2, 0.6, .2, .2], facecolor='k')
plt.plot(t[:len(r)], r)
plt.title('Impulse response')
plt.xlim(0, 0.2)
plt.xticks([])
plt.yticks([])

plt.show()

# Meins
plt.figure(1, figsize=(9, 3))
plt.subplot(1, 1, 1)
plt.plot([1, 2], [3, 2], color='black')
ax2 = plt.axes([0, 0, 1, 1], facecolor='none')
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.plot([1, 2], [1, 1])

# Linien
plt.axhline(1)
plt.axhline(2)

'''
Tick-Labels 
'''
x = [1, 2, 3, 4]
y = [1, 4, 9, 6]
labels = ['Frogs', 'Hogs', 'Bogs', 'Slogs']

plt.plot(x, y)  # You can specify a rotation for the tick labels in degrees or with keywords.
plt.xticks(x, labels, rotation='vertical')
plt.margins(0.2)  # Pad margins so that markers don't get clipped by the axes
plt.subplots_adjust(bottom=0.15)  # Tweak spacing to prevent clipping of tick-labels
plt.show()

# Custom tick labels
x = [0, 5, 9, 10, 15]
y = [0, 1, 2, 3, 4]
fig, ax = plt.subplots()
ax.plot(x, y)
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 0.712123))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
plt.show()

'''

'''

##
# Working with text
###
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the dat
n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

# %%
# 3D

# Scatter
a = 2
x = np.repeat(np.arange(0, 3, 0.01), 300)
y = np.tile(np.arange(0, 3, 0.01), 300)
f = lambda x, y: x * y / (2 * a + x) ** 2
z = f(x, y)
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z, c=z, cmap='Reds');

# Line
x = np.arange(0, 3, 0.01)
y = x
z = (x + y) / 1000
ax.plot3D(x, y, z, 'gray')




#%%
"""
# Colors
"""
for name, hex in matplotlib.colors.cnames.items():
    print(name, hex)
