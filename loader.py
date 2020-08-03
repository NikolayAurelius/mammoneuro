import pickle
import torch
import numpy as np
import os
from common import MammographMatrix

# torch.rot90


class Loader:
    def __init__(self, dataset_path='dataset', part='train'):
        self.mammograph_matrix = MammographMatrix().matrix

        self.dataset_path = dataset_path
        self.txt_filenames = os.listdir(f'{self.dataset_path}/txt_files')
        with open(f'{self.dataset_path}/target_by_filename.pickle', 'rb') as f:
            self.markup = pickle.load(f)
        self.txt_filenames = list(set(self.txt_filenames).intersection(self.markup.keys()))

        self.dataset_length = len(self.txt_filenames)
        print(f'Найдено обучающих примеров: {self.dataset_length}')

        self.split(part)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = {'X': torch.zeros((self.part_length, 1, 18, 18, 18, 18), device=self.device),
                        'Y': torch.zeros((self.part_length, 1), device=self.device)}

        self.load(normalize=True)
        self.work_mode()

    def work_mode(self):
        del self.mammograph_matrix, self.dataset_length, self.txt_filenames
        del self.dataset_path, self.part_markup, self.device, self.markup

    def split(self, part):
        positive_filenames = list([elem for elem in self.txt_filenames if self.markup[elem]])
        negative_filenames = list([elem for elem in self.txt_filenames if not self.markup[elem]])
        print(f'Positive: {len(positive_filenames)} Negative: {len(negative_filenames)} '
              f'Relation: {len(positive_filenames) / len(self.txt_filenames)}')

        np.random.seed(17021999)
        np.random.shuffle(positive_filenames)
        np.random.shuffle(negative_filenames)

        part_filenames = []
        if part == 'train':
            b = int(len(positive_filenames) * 0.7)
            d = int(len(negative_filenames) * 0.7)
            part_filenames.extend(positive_filenames[:b])
            part_filenames.extend(negative_filenames[:d])

        elif part == 'val':
            a, b = int(len(positive_filenames) * 0.7), int(len(positive_filenames) * 0.9)
            c, d = int(len(negative_filenames) * 0.7), int(len(negative_filenames) * 0.9)
            part_filenames.extend(positive_filenames[a:b])
            part_filenames.extend(negative_filenames[c:d])
        elif part == 'test':
            a = int(len(positive_filenames) * 0.9)
            c = int(len(negative_filenames) * 0.9)
            part_filenames.extend(positive_filenames[a:])
            part_filenames.extend(negative_filenames[c:])
        elif part == 'val+test':
            a = int(len(positive_filenames) * 0.7)
            c = int(len(negative_filenames) * 0.7)
            part_filenames.extend(positive_filenames[a:])
            part_filenames.extend(negative_filenames[c:])
        else:
            raise ValueError(f'check "part" argument')
        self.part_length = len(part_filenames)

        print(f'Part: {part} Количество: {self.part_length}')
        self.part_markup = {key: self.markup[key] for key in part_filenames}

    def txt_file_to_x(self, path):
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

    def load(self, normalize:bool = False):
        self.dataset_filenames = np.array(list(self.part_markup.keys()))
        for index, filename in enumerate(self.dataset_filenames):
            x = self.txt_file_to_x(f'{self.dataset_path}/txt_files/{filename}')
            y = [int(self.part_markup[filename])]

            self.dataset['X'][index] = torch.Tensor(x).type(torch.FloatTensor).to(device=self.device)
            self.dataset['Y'][index] = torch.Tensor(y).type(torch.LongTensor).to(device=self.device)

            if normalize:
                self.dataset['X'][index] = self.dataset['X'][index] / torch.max(self.dataset['X'][index])

    def generator(self, batch_size=64):
        batch_size = min(batch_size, self.part_length)
        indexes = list(range(self.part_length))
        while True:
            np.random.shuffle(indexes)

            for step in range(self.part_length // batch_size):
                curr_indexes = indexes[batch_size * step: batch_size * (step + 1)]
                yield self.dataset_filenames[curr_indexes], self.dataset['X'][curr_indexes], self.dataset['Y'][curr_indexes]
