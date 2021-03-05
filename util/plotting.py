import matplotlib.pyplot as plt


def show_histogram(series, title):
    fig, ax = plt.subplots()
    ax.hist(series)
    ax.set_title(title)
    plt.show()
