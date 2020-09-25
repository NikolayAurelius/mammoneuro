import pickle
import torch
import numpy as np
import os
from common import MammographMatrix


class Loader:
    def __init__(self, dataset_path: str = 'dataset', tosplit = None):

        self.mammograph_matrix = MammographMatrix().matrix 

        self.dataset_path = dataset_path
        self.txt_filenames = os.listdir(f'{self.dataset_path}/txt_files')

        with open(f'{self.dataset_path}/markup_with_info.pickle', 'rb') as f:
            self.markup = pickle.load(f)

        self.txt_filenames = list(set(self.txt_filenames).intersection(self.markup.keys()))
 
        self.dataset_length = len(self.txt_filenames)
        print(f'Найдено обучающих примеров: {self.dataset_length}')
 
        self.split(tosplit)
 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = {'X': torch.zeros((self.part_length, 1, 18, 18, 18, 18), device=self.device),
                        'Y': torch.zeros((self.part_length, 1), device=self.device)}
 
        self.load(normalize=True)
        self.work_mode()
 
    def work_mode(self):
        del self.mammograph_matrix, self.dataset_length, self.txt_filenames
        del self.dataset_path, self.part_markup, self.device, self.markup
 

    def split(self, tosplit = (1., 1.)):

        if tosplit:
        
          target_rltn, side_rltn = tosplit

          istarg = 1
          isnottarg = 1
          isleft = 1
          isright = 1
          
          self.part_filenames = []

          n = 5

          for elem in self.txt_filenames:

            if self.markup[elem]['target']:

              if self.markup[elem]['side'] == 'Левая':
                
                if isleft / (isright + n) <= side_rltn:
                  if istarg / (isnottarg + n) <= target_rltn:

                    self.part_filenames.append(elem)
                    istarg += 1
                    isleft += 1

              else:

                if (isleft + n) / isright > side_rltn:
                  if istarg / (isnottarg + n) <= target_rltn:

                    self.part_filenames.append(elem)
                    istarg += 1
                    isright += 1
              
            else:
            
              if self.markup[elem]['side'] == 'Левая': 

                if isleft / (isright + n) <= side_rltn:
                  if (istarg + n) / isnottarg > target_rltn:

                    self.part_filenames.append(elem)
                    isnottarg += 1
                    isleft += 1

              else:

                if (isleft + n) / isright > side_rltn:
                  if (istarg + n) / isnottarg > target_rltn:

                    self.part_filenames.append(elem)
                    isnottarg += 1
                    isright += 1

        else:
          self.part_filenames = self.txt_filenames

        np.random.seed(17021999)
        np.random.shuffle(self.part_filenames)
        self.part_length = len(self.part_filenames)
        print(f'Количество: {self.part_length}')
        self.part_markup = {key: self.markup[key] for key in self.part_filenames}

    def txt_file_to_x(self, path: str):
        with open(path, encoding='cp1251') as f:
            need_check = True
            lst = []
            for line in f.readlines():
                if need_check and line.count('0;') != 0:
                    need_check = False
                elif not need_check:
                    pass
                else:
                    continue
 
                one_x = np.zeros((18, 18))
                line = line[:-2].split(';')
 
                for i in range(18):
                    for j in range(18):
                        one_x[i, j] = int(line[i * 18 + j])
                lst.append(one_x)
 
            x = np.zeros((18, 18, 18, 18))
 
            for i in range(18):
                for j in range(18):
                    if self.mammograph_matrix[i, j] != -1:
                        x[i, j] = lst[self.mammograph_matrix[i, j] - 1]
        return x
 
    def load(self, normalize: bool = False):
        self.dataset_filenames = np.array(list(self.part_markup.keys()))
        for index, filename in enumerate(self.dataset_filenames):
            x = self.txt_file_to_x(f'{self.dataset_path}/txt_files/{filename}')
            y = [int(self.part_markup[filename]['target'])]
 
            self.dataset['X'][index] = torch.Tensor(x).type(torch.FloatTensor).to(device=self.device)
            self.dataset['Y'][index] = torch.Tensor(y).type(torch.LongTensor).to(device=self.device)
 
            if normalize:
                self.dataset['X'][index] = self.dataset['X'][index] / torch.max(self.dataset['X'][index])
 
    def generator(self, batch_size: int = 64):
        batch_size = min(batch_size, self.part_length)
        indexes = list(range(self.part_length))
        while True:
            np.random.shuffle(indexes)
 
            for step in range(self.part_length // batch_size):
                curr_indexes = indexes[batch_size * step: batch_size * (step + 1)]
                yield self.dataset_filenames[curr_indexes], self.dataset['X'][curr_indexes], self.dataset['Y'][curr_indexes]


class AmplAugmentator:

    def __init__(self, loader):
        self.loader = loader

    def meas_to_x(self, meas: np.array = np.zeros((18, 18, 18, 18, 80)), w=10000., wd=300000.):

        t_meas = torch.tensor(meas).float()
        smeas, _ = torch.sort(t_meas, axis=4)
        mxmn = torch.tensor((smeas[:, :, :, :, -1] - smeas[:, :, :, :, 0]) / 2, requires_grad=False)
        ones = torch.ones_like(mxmn)

        A = torch.tensor((smeas[:, :, :, :, -1] - smeas[:, :, :, :, 0]) / 2, requires_grad=True)
        w = torch.tensor(w)
        epsilon = torch.tensor(torch.zeros_like(A), requires_grad=True)
        b = torch.tensor(torch.mean(t_meas, axis=4), requires_grad=True)
        wd = torch.tensor(wd)

        t = lambda i: t(i - 1) + (1. / wd) if i > 0 else i
        y = lambda i: (torch.abs(A) * torch.sin(2 * np.pi * w * ones * t(i) + epsilon) + b) * (mxmn != 0.0)

        optimizer = torch.optim.SGD(
            [{'params': [epsilon], 'lr': 0.01}, {'params': [A], 'lr': 100000}, {'params': [b], 'lr': 10000}])

        loss = meas.shape[-1] * 40 + 1
        k = 0
        while loss > (meas.shape[-1] * 40):
            # for k in range(2000):

            optimizer.zero_grad()
            loss = 0.0

            for i in range(0, meas.shape[-1]):
                target = t_meas[:, :, :, :, i]
                y_pred = y(i)
                vloss = torch.mean(torch.abs(y_pred - target))
                vloss.backward()
                loss += vloss

            optimizer.step()

            if k % 500 == 0:
                print(meas.shape[-1])
                print('Iteration: ', k)
                print('Loss: ', loss.item())

            k += 1

        return A.detach().numpy()

    def idxs_generator(self, length):
        return random.sample([i for i in range(80)], length)

    def get_new_amplitude(self, meas, dots_kolvo=3):

        new_meas = np.zeros((18, 18, 18, 18, dots_kolvo))

        for i in range(18):
            for j in range(18):
                for k in range(18):
                    for l in range(18):
                        sin = meas[i, j, k, l]
                        new_meas[i, j, k, l] = sin[idxs_generator(dots_kolvo)]

        return self.meas_to_x(new_meas)

    def generator(self, batch_size: int = 16):
        for filename, x, target in self.loader.generator(batch_size):
            yield filename, self.get_new_amplitude(x), target