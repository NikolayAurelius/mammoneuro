import pickle
import torch
import numpy as np
import os


# torch.rot90

class Loader:
    def __init__(self, dataset_path='dataset', part='train'):
        self.mammograph_matrix = np.zeros((18, 18), dtype=np.int32)
        self.init_mammograph_matrix()
        self.dataset_path = dataset_path
        self.txt_filenames = os.listdir(f'{self.dataset_path}/txt_files')
        with open(f'{self.dataset_path}/target_by_filename.pickle', 'rb') as f:
            self.markup = pickle.load(f)
        self.txt_filenames = list(set(self.txt_filenames).intersection(self.markup.keys()))

        self.dataset_length = len(self.txt_filenames)
        print(f'Найдено обучающих примеров: {self.dataset_length}')

        filenames = list(self.txt_filenames)
        np.random.seed(17021999)
        np.random.shuffle(filenames)

        part_filenames = []
        if part == 'train':
            self.part_length = int(self.dataset_length * 0.7)
            part_filenames.extend(filenames[:int(self.dataset_length * 0.7)])
        elif part == 'val':
            self.part_length = int(self.dataset_length * 0.9) - int(self.dataset_length * 0.7)
            part_filenames.extend(filenames[int(self.dataset_length * 0.7):int(self.dataset_length * 0.9)])
        elif part == 'test':
            self.part_length = self.dataset_length - int(self.dataset_length * 0.9)
            part_filenames.extend(filenames[int(self.dataset_length * 0.9):])
        elif part == 'val+test':
            self.part_length = self.dataset_length - int(self.dataset_length * 0.7)
            part_filenames.extend(filenames[int(self.dataset_length * 0.7):])
        else:
            raise ValueError(f'check "part" argument')

        print(f'Part: {part} Количество: {self.part_length}')
        self.part_markup = {key: self.markup[key] for key in part_filenames}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = {'X': torch.zeros((self.part_length, 1, 18, 18, 18, 18), device=self.device),
                        'Y': torch.zeros((self.part_length, 1), device=self.device)}

        self.load()

    def init_mammograph_matrix(self):
        self.mammograph_matrix = self.mammograph_matrix - 1
        gen = iter(range(256))

        for i in range(6, 18 - 6):
            self.mammograph_matrix[0, i] = next(gen)

        for i in range(4, 18 - 4):
            self.mammograph_matrix[1, i] = next(gen)

        for i in range(3, 18 - 3):
            self.mammograph_matrix[2, i] = next(gen)

        for i in range(2, 18 - 2):
            self.mammograph_matrix[3, i] = next(gen)

        for j in range(2):
            for i in range(1, 18 - 1):
                self.mammograph_matrix[4 + j, i] = next(gen)

        for j in range(6):
            for i in range(18):
                self.mammograph_matrix[6 + j, i] = next(gen)

        for j in range(2):
            for i in range(1, 18 - 1):
                self.mammograph_matrix[12 + j, i] = next(gen)

        for i in range(2, 18 - 2):
            self.mammograph_matrix[14, i] = next(gen)

        for i in range(3, 18 - 3):
            self.mammograph_matrix[15, i] = next(gen)

        for i in range(4, 18 - 4):
            self.mammograph_matrix[16, i] = next(gen)

        for i in range(6, 18 - 6):
            self.mammograph_matrix[17, i] = next(gen)

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

    def load(self):
        for index, filename in enumerate(self.part_markup.keys()):
            x = self.txt_file_to_x(f'{self.dataset_path}/txt_files/{filename}')
            y = [int(self.part_markup[filename])]

            self.dataset['X'][index] = torch.Tensor(x).type(torch.FloatTensor).to(device=self.device)
            self.dataset['Y'][index] = torch.Tensor(y).type(torch.LongTensor).to(device=self.device)

    def generator(self, batch_size=64):
        batch_size = min(batch_size, self.part_length)
        indexes = list(range(self.part_length))
        while True:
            np.random.shuffle(indexes)

            for step in range(self.part_length // batch_size):
                curr_indexes = indexes[batch_size * step: batch_size * (step + 1)]
                yield self.dataset['X'][curr_indexes], self.dataset['Y'][curr_indexes]
