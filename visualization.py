import enum
from time import time
from mpl_toolkits.mplot3d import Axes3D

from operator import mod
from statistics import mode
import xdrlib
from utils import create_random_direction, get_weights, eval_loss,test
import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import cm
from tqdm import tqdm




def get_direction(model,weight, random=True):
    if random:
        xdirection = create_random_direction(model, weight)
        ydirection = create_random_direction(model, weight)
    return [xdirection, ydirection]


def plot_contour_trajectory(val, xcoord_mesh, ycoord_mesh, vmin=0.1, vmax=10, vlevel=0.5):
    fig = plt.figure()
    CS = plt.contour(xcoord_mesh, ycoord_mesh, val, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig('2dcontour' + '.pdf', dpi=300, bbox_inches='tight', format='pdf')

    fig = plt.figure()
    CS = plt.contourf(xcoord_mesh, ycoord_mesh, val, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    fig.savefig('2dcontourf' + '.pdf', dpi=300, bbox_inches='tight', format='pdf')

    fig = plt.figure()
    sns_plot = sns.heatmap(val, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig('2dheat.pdf', dpi=300, bbox_inches='tight', format='pdf')



    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(xcoord_mesh, ycoord_mesh, val, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig('3dsurface.pdf', dpi=300,
                bbox_inches='tight', format='pdf')
    pass


def plot(model,weight,dataloader,criterion, xcoordinates, ycoordinates):
    shape = (len(xcoordinates),len(ycoordinates))
    losses = -np.ones(shape=shape)
    accuracies = -np.ones(shape=shape)
    inds = np.array(range(losses.size))
    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
    s1 = xcoord_mesh.ravel()
    s2 = ycoord_mesh.ravel()
    coords = np.c_[s1,s2]
    direction = get_direction(model,weight, random=True)
    print('begin cal')

    for count, ind in enumerate(tqdm(inds)):
        coord = coords[count]
        dx = direction[0]
        dy = direction[1]
        changes = [d0*coord[0] + d1*coord[1] for (d0, d1) in zip(dx, dy)]
        for (p, w, d) in zip(model.parameters(), weight, changes):
            p.data = w + torch.Tensor(d).type(type(w))
        stime = time()
        acc, loss = test(model, dataloader, criterion)
        # print('cost: ', stime-time())
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc
        # loss, acc = eval_loss(model, criterion, dataloader)
    print('-------------')
    plot_contour_trajectory(losses, xcoord_mesh, ycoord_mesh)
    plot_contour_trajectory(accuracies, xcoord_mesh, ycoord_mesh)






