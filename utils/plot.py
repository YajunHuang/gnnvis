import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_and_save_data(embeddings, labels, name):
    '''
    Save and print a scatter plot
    :param embeddings:
    :param labels:
    :param name:
    :return:
    '''
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    c = labels

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax1.scatter(x, y, c=c, s=2)
    plt.savefig("./result/" + name, dpi = 800)


def scatter(embeddings, labels, fig_path, fig_range=None):
    all_data = {}
    for i in range(len(labels)):
        if labels[i] in all_data:
            all_data[labels[i]].append(embeddings[i])
        else:
            all_data[labels[i]] = [embeddings[i]]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(all_data)))
    ll_color_handles = []
    for color, ll in zip(colors, sorted(all_data.keys())):
        x = [t[0] for t in all_data[ll]]
        y = [t[1] for t in all_data[ll]]
        plt.plot(x, y, '.', color = color, label = ll, markersize = 1)

    if len(all_data.keys()) > 1:
        ncol = int(len(all_data.keys()) / 2. + 0.5)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=10, mode="expand", borderaxespad=0.)

    if fig_range != None and fig_range != '':
        l = abs(float(fig_range))
        plt.xlim(-l, l)
        plt.ylim(-l, l)
    plt.savefig(fig_path, dpi = 500)
    plt.clf()


def hist2d(embeddings, labels, fig_fname, fig_range=None):
    all_data = {}
    for i in range(len(labels)):
        if labels[i] in all_data:
            all_data[labels[i]].append(embeddings[i])
        else:
            all_data[labels[i]] = [embeddings[i]]
    x = []
    y = []
    for ll in sorted(all_data.keys()):
        x += [t[0] for t in all_data[ll]]
        y += [t[1] for t in all_data[ll]]
    plt.hist2d(x, y, bins=(500, 500), norm=LogNorm())
    plt.colorbar()
    if fig_range != None and fig_range != '':
        l = abs(float(fig_range))
        plt.xlim(-l, l)
        plt.ylim(-l, l)
    plt.savefig("./result/" + fig_fname, dpi = 500)
    plt.clf()


def save_plot_data(x, y, path):
    dic = {}
    dic['x'] = x
    dic['y'] = y
    with open('./result/visdata/'+path, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_plot_data(path):
    with open('./result/visdata/'+path, 'rb') as handle:
        b = pickle.load(handle)
        x = b['x']
        y = b['y']
    return x, y