import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import seaborn as sns
sns.set(color_codes=True)

### -------------- гифка с распределениями во времени ---------- ###


def make_distplot_gif(lst0, lst1, filename = 'distplotsgif'):

    # lst0, lst1 - первый и второй список списков предсказаний

    fig, im = plt.subplots()

    im = sns.distplot(lst0[0], color = 'r', label="iscat")
    im = sns.distplot(lst1[0], color = 'b', ax = im, label="isdog")
    im.axes.set_title("Iteration: " + '1')
    im.set(xlim = (-20,20))
    im.set(ylim = (0,0.5))
    plt.legend()


    def update(i):
        fig.clear()
        im = sns.distplot(lst0[i], color = 'r', label="iscat")
        im = sns.distplot(lst1[i], color = 'b', ax = im, label="isdog")
        im.axes.set_title("Iteration: " + str(i+1))
        im.set(xlim = (-20,20))
        im.set(ylim = (0,0.5))
        plt.legend()
        return im

    anim = FuncAnimation(fig, update, frames=range(len(lst0)), interval=300)
    anim.save(filename + '.gif', dpi=80, writer='pillow')

### -------------- 3д графики с распределениями от времени ---------- ###


def dist_to_3d(lists1, lists2, xdiap = None, ydiap = None, toSave = False, filename = '3ddists'):

  verts1 = []
  verts2 = []

  for i in range(len(lists1)):
    fig, im = plt.subplots(figsize = (15, 15))
    x, y = sns.distplot(lists1[i], color = 'r', ax = im).get_lines()[0].get_data()
    plt.close()
    verts1.append(list(zip(x, y)))

    fig, im = plt.subplots()
    x, y = sns.distplot(lists2[i], color = 'r', ax = im).get_lines()[0].get_data()
    plt.close()
    verts2.append(list(zip(x, y)))

  fig = plt.figure()
  ax = fig.gca(projection='3d')

  zs = np.linspace(0, 2, len(lists1))

  if (xdiap):
    ax.set_xlim3d(xdiap[0], xdiap[1])
  else: 
    ax.set_xlim3d(min(x), max(x))

  ax.set_ylim3d(-0.5, 2.5)
  
  if (ydiap):
    ax.set_zlim3d (ydiap[0], ydiap[1])
  else: 
    ax.set_zlim3d(min(y), max(y))

  ax.set_xlabel('probability')
  ax.set_ylabel('epochs')
  ax.set_zlabel('distribution')

  # первые распределения
  poly = PolyCollection(verts1, facecolors = ['r'])
  poly.set_alpha(0.7)

  # вторые распределения
  poly2 = PolyCollection(verts2, facecolors = ['b'])
  poly2.set_alpha(0.7)

  ax.add_collection3d(poly2, zs=zs, zdir='y')

  ax.add_collection3d(poly, zs=zs, zdir='y')

  if (toSave):
    plt.savefig(filename)
  else:
    plt.show()



  ### -------------- distribution_of_array ---------- ###

def dist_of_array(X, Y, toSave = False, filename = 'dist_of_array'):

  fig, ax = plt.subplots(figsize=(30, 18))

  new_X = []

  for i in range(len(X)):
    new_X.append(X[i][0])
  
  new_X = np.array(new_X)

  plain_X = list()

  for k in range(new_X.shape[0]):
    plain_X.append(np.reshape(new_X[k], 18 ** 4))


  a = []
  b = []
  for index in range(len(plain_X)):
    x, y = plain_X[index], Y[index]
    x += np.abs(np.min(x))

    x = np.log(x + 1)

    if y[0] == 0.0:
        a.append(x)
    else:
        b.append(x)

  x1 = a[0]
  for elem in a[1:]:
    x1 = np.concatenate((x1, elem), axis=0)

  im = sns.distplot(x1, bins=None, ax = ax, color = 'r')

  x2 = b[0]
  for elem in b[1:]:
    x2 = np.concatenate((x2, elem), axis=0)

  sns.distplot(x2, bins=None, ax = ax, color = 'b')

  if (toSave):
    plt.savefig(filename)
  else:
    plt.show()
  