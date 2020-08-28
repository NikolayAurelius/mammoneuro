from vizualization_stuff.distplots import make_distplot_gif, dist_to_3d, dist_of_array
from vizualization_stuff.graph_3d import make_my_plot
from vizualization_stuff.analytics_of_errors import sub_deviance, deviance, matrix_voltage_error, sub_meas, meas, sub_distance_meas

from common import Loader, MammographMatrix

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

def make_adjancecy_mx(X):
  M = np.zeros((18 * 18, 18 * 18))
  for k in range(18):
      for l in range(18):
          for i in range(18):
              for j in range(18):
                  M[18 * k + l][18 * i + j] = X[k][l][i][j]
  return M

def show_plots(vals, toSave = False, filename = 'plots_of_matriсes'):


  fig = plt.figure(figsize=(15,10))

  grid = AxesGrid(fig, 111,
                nrows_ncols=(1, len(vals)),
                axes_pad=0.05,
                share_all=True,
                label_mode="L",
                cbar_location="right",
                cbar_mode="single",
                )

  for val, ax in zip(vals,grid):
      im = ax.imshow(val)

  grid.cbar_axes[0].colorbar(im)

  for cax in grid.cbar_axes:
    cax.toggle_label(False)


  if (toSave):
    plt.savefig(filename)
  else:
    plt.show()


if __name__ == '__main__':

  # Загружаем снимки и target

  loader = Loader(dataset_path='dataset')
  _, x, y = next(loader.generator(50))
  x = x.cpu().detach().numpy()

  ### -------------- distribution_of_array ---------- ###

  #dist_of_array(x, y, toSave = True, filename = 'vizualization_stuff/pics/dist_of_array')

  ### -------------- 3д граф ---------- ###
  
  x = x[0, 0]
  #make_my_plot(x, figursize = (20,20), toSave = True, 
  #  filename = vizualization_stuff/pics/3dplot', euclid_colors = True)



  ### -------------- разные аналитические матрицы -------------- ###

  img0 = np.zeros((18, 18))
  for i in range(18):
    for j in range(18):
        img0[i, j] = x[i, j, i, j]
  img1 = deviance(x, 1, MammographMatrix)
  img2 = meas(x, 1, MammographMatrix)
  img3 = matrix_voltage_error(x, MammographMatrix)

  #show_plots([img0, img1, img2, img3], toSave = True,  filename = 'vizualization_stuff/pics/matrixes')




  ### -------------- гифка и 3д график распределений ---------- ###

  #make_distplot_gif(lst0, lst1,  filename = 'vizualization_stuff/pics/distplotsgif') 

  #dist_to_3d(lst0, lst1, toSave = True, filename = 'vizualization_stuff/pics/3ddists')
