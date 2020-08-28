import numpy as np
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

### --------------большой граф с ребрами ---------- ###

def make_my_plot(X, node_color = 'black', 
	edge_alpha = 0.2, node_alpha = 0.1, 
	figursize = (10,10), toSave = False, 
  filename = 'myplot', euclid_colors = False):

  # на вход подается замер

  N = 18*18
  # построим матрицу смежности
  M = np.zeros((18 * 18, 18 * 18))
  for k in range(18):
      for l in range(18):
          for i in range(18):
              for j in range(18):
                  M[18 * k + l][18 * i + j] = X[k][l][i][j]

  colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

  fig = plt.figure(figsize=figursize)
  ax = Axes3D(fig)
  grid_size = math.ceil(math.sqrt(N)) - 1 + 1*(N == 1)

  # составим матрицу координат (расставим точки в узлах сетки)
  N_coords = []

  itr = 0
  for i in range (grid_size + 1):
    for j in range (grid_size + 1):
      N_coords.append((i, j))
      itr += 1
      if (itr == N):
        break
    if (itr == N):
      break

  # границы графика
  ax.set_xlim3d(0, grid_size)
  ax.set_ylim3d(0, grid_size)
  ax.set_zlim3d(0, M.max())

  for note in N_coords:
    ax.scatter(note[0], note[1], 0, c=node_color, alpha = node_alpha)

  M_edges = M.nonzero() # получаем два списка где смежные точки соответствуют друг другу поэлементно
  for i in range(len(M_edges[0])): #перебираем все точки
    x0 = N_coords[ M_edges[0][i] ][0]
    x1 = N_coords[ M_edges[1][i] ][0]
    y0 = N_coords[ M_edges[0][i] ][1]
    y1 = N_coords[ M_edges[1][i] ][1]
    weight = M[M_edges[0][i]][M_edges[1][i]]
    x = np.linspace(x0, x1, 10) 
    y = np.linspace(y0, y1, 10) 
    euc = math.sqrt(((x1 - x0) ** 2 + (y1 - y0) ** 2))/math.sqrt((grid_size**2 + grid_size**2))
    if (euclid_colors):
      edge_col = colors[int(euc*7)]
    else: 
      edge_col = 'black'
    z = (1 - np.linspace(-1,1,10)**2) * weight
    ax.plot(x,y,z, c = edge_col, alpha = edge_alpha)
    ax.scatter(x1, y1, 0, c = edge_col, alpha = edge_alpha, 
    	s = 50, marker=(3, 0, 200))
  
  if (toSave):
    plt.savefig(filename)
  else:
    plt.show()
