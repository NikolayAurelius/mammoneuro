import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Callable, Union
import math
import numpy as np
import os
import pickle
from mammo_packets import read_from_file_binary, parse_mammograph_raw_data


class Loader:
    def __init__(self, dataset_path: str = 'dataset', part: str = 'train', lst: list = [],
                 normalize: bool = True, normalize_func=lambda x: x):
        self.mammograph_matrix = MammographMatrix().matrix

        self.dataset_path = dataset_path
        self.txt_filenames = os.listdir(f'{self.dataset_path}/txt_files')
        with open(f'{self.dataset_path}/target_by_filename.pickle', 'rb') as f:
            self.markup = pickle.load(f)

        self.txt_filenames = list(set(self.txt_filenames).intersection(self.markup.keys()))

        if len(lst) != 0:
            self.txt_filenames = list(set(self.txt_filenames).intersection(set(lst)))

        self.dataset_length = len(self.txt_filenames)
        print(f'Найдено обучающих примеров: {self.dataset_length}')

        self.split(part)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = {'X': torch.zeros((self.part_length, 1, 18, 18, 18, 18), device=self.device),
                        'Y': torch.zeros((self.part_length, 1), device=self.device)}

        self.normalize_func = normalize_func
        self.load(normalize=normalize)
        self.work_mode()

    def work_mode(self):
        del self.mammograph_matrix, self.dataset_length, self.txt_filenames
        del self.dataset_path, self.part_markup, self.device, self.markup

    def split(self, part: str):
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
        elif part == 'all':
            part_filenames.extend(positive_filenames)
            part_filenames.extend(negative_filenames)
        else:
            raise ValueError(f'check "part" argument')
        self.part_length = len(part_filenames)

        print(f'Part: {part} Количество: {self.part_length}')
        self.part_markup = {key: self.markup[key] for key in part_filenames}

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
            y = [int(self.part_markup[filename])]

            self.dataset['X'][index] = torch.Tensor(x).type(torch.FloatTensor).to(device=self.device)
            self.dataset['Y'][index] = torch.Tensor(y).type(torch.LongTensor).to(device=self.device)

            if normalize:
                self.dataset['X'][index] = self.normalize_func(self.dataset['X'][index])

    def generator(self, batch_size: int = 64):
        batch_size = min(batch_size, self.part_length)
        indexes = list(range(self.part_length))
        while True:
            np.random.shuffle(indexes)

            for step in range(self.part_length // batch_size):
                curr_indexes = indexes[batch_size * step: batch_size * (step + 1)]
                yield self.dataset_filenames[curr_indexes], self.dataset['X'][curr_indexes], self.dataset['Y'][curr_indexes]

    def aug_generator(self, batch_size: int = 16, need_concatenate: bool = True):
        self.augmentator = Augmentator(self)
        return self.augmentator.generator(batch_size, need_concatenate)


class ConvNd(nn.Module):
    """Some Information about convNd"""

    def __init__(self, in_channels: int,
                 out_channels: int,
                 num_dims: int,
                 kernel_size: Tuple,
                 stride: Union[int, Tuple],
                 padding: Union[int, Tuple],
                 is_transposed: bool = False,
                 padding_mode='zeros',
                 output_padding=0,
                 dilation: int = 1,
                 groups: int = 1,
                 rank: int = 0,
                 use_bias: bool = True,
                 bias_initializer: Callable = None,
                 kernel_initializer: Callable = None):
        super(ConvNd, self).__init__()

        # ---------------------------------------------------------------------
        # Assertions for constructor arguments
        # ---------------------------------------------------------------------
        if not isinstance(kernel_size, Tuple):
            kernel_size = tuple(kernel_size for _ in range(num_dims))
        if not isinstance(stride, Tuple):
            stride = tuple(stride for _ in range(num_dims))
        if not isinstance(padding, Tuple):
            padding = tuple(padding for _ in range(num_dims))
        if not isinstance(output_padding, Tuple):
            output_padding = tuple(output_padding for _ in range(num_dims))
        if not isinstance(dilation, Tuple):
            dilation = tuple(dilation for _ in range(num_dims))

        # This parameter defines which Pytorch convolution to use as a base, for 3 Conv2D is used
        if rank == 0 and num_dims <= 3:
            max_dims = num_dims - 1
        else:
            max_dims = 3

        if is_transposed:
            self.conv_f = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)[max_dims - 1]
        else:
            self.conv_f = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[max_dims - 1]

        assert len(kernel_size) == num_dims, \
            'nD kernel size expected!'
        assert len(stride) == num_dims, \
            'nD stride size expected!'
        assert len(padding) == num_dims, \
            'nD padding size expected!'
        assert len(output_padding) == num_dims, \
            'nD output_padding size expected!'
        assert sum(dilation) == num_dims, \
            'Dilation rate other than 1 not yet implemented!'

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------
        self.rank = rank
        self.is_transposed = is_transposed
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_dims = num_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        # ---------------------------------------------------------------------
        # Construct 3D convolutional layers
        # ---------------------------------------------------------------------
        if self.bias_initializer is not None:
            if self.use_bias:
                self.bias_initializer(self.bias)
        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv_layers = torch.nn.ModuleList()

        # Compute the next dimension, so for a conv4D, get index 3
        next_dim_len = self.kernel_size[0]

        for _ in range(next_dim_len):
            if self.num_dims - 1 > max_dims:
                # Initialize a Conv_n-1_D layer
                conv_layer = ConvNd(in_channels=self.in_channels,
                                    out_channels=self.out_channels,
                                    use_bias=self.use_bias,
                                    num_dims=self.num_dims - 1,
                                    rank=self.rank - 1,
                                    is_transposed=self.is_transposed,
                                    kernel_size=self.kernel_size[1:],
                                    stride=self.stride[1:],
                                    groups=self.groups,
                                    dilation=self.dilation[1:],
                                    padding=self.padding[1:],
                                    padding_mode=self.padding_mode,
                                    output_padding=self.output_padding[1:],
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer)

            else:
                # Initialize a Conv layer
                # bias should only be applied by the top most layer, so we disable bias in the internal convs
                conv_layer = self.conv_f(in_channels=self.in_channels,
                                         out_channels=self.out_channels,
                                         bias=False,
                                         kernel_size=self.kernel_size[1:],
                                         dilation=self.dilation[1:],
                                         stride=self.stride[1:],
                                         padding=self.padding[1:],
                                         padding_mode=self.padding_mode,
                                         groups=self.groups)
                if self.is_transposed:
                    conv_layer.output_padding = self.output_padding[1:]

                # Apply initializer functions to weight and bias tensor
                if self.kernel_initializer is not None:
                    self.kernel_initializer(conv_layer.weight)

            # Store the layer
            self.conv_layers.append(conv_layer)

    # -------------------------------------------------------------------------

    def forward(self, input):

        # Pad the input if is not transposed convolution
        if not self.is_transposed:
            padding = list(self.padding)
            # Pad input if this is the parent convolution ie rank=0
            if self.rank == 0:
                inputShape = list(input.shape)
                inputShape[2] += 2 * self.padding[0]
                padSize = (0, 0, self.padding[0], self.padding[0])
                padding[0] = 0
                if self.padding_mode is 'zeros':
                    input = F.pad(input.view(input.shape[0], input.shape[1], input.shape[2], -1), padSize, 'constant',
                                  0).view(inputShape)
                else:
                    input = F.pad(input.view(input.shape[0], input.shape[1], input.shape[2], -1), padSize,
                                  self.padding_mode).view(inputShape)

        # Define shortcut names for dimensions of input and kernel
        (b, c_i) = tuple(input.shape[0:2])
        size_i = tuple(input.shape[2:])
        size_k = self.kernel_size

        if not self.is_transposed:
            # Compute the size of the output tensor based on the zero padding
            size_o = tuple(
                [math.floor((size_i[x] + 2 * padding[x] - size_k[x]) / self.stride[x] + 1) for x in range(len(size_i))])
            # Compute size of the output without stride
            size_ons = tuple([size_i[x] - size_k[x] + 1 for x in range(len(size_i))])
        else:
            # Compute the size of the output tensor based on the zero padding
            size_o = tuple(
                [(size_i[x] - 1) * self.stride[x] - 2 * self.padding[x] + (size_k[x] - 1) + 1 + self.output_padding[x]
                 for x in range(len(size_i))])

        # Output tensors for each 3D frame
        frame_results = size_o[0] * [torch.zeros((b, self.out_channels) + size_o[1:], device=input.device)]
        empty_frames = size_o[0] * [None]

        # Convolve each kernel frame i with each input frame j
        for i in range(size_k[0]):
            # iterate inputs first dimmension
            for j in range(size_i[0]):

                # Add results to this output frame
                if self.is_transposed:
                    out_frame = i + j * self.stride[0] - self.padding[0]
                else:
                    out_frame = j - (i - size_k[0] // 2) - (size_i[0] - size_ons[0]) // 2 - (1 - size_k[0] % 2)
                    k_center_position = out_frame % self.stride[0]
                    out_frame = math.floor(out_frame / self.stride[0])
                    if k_center_position != 0:
                        continue

                if out_frame < 0 or out_frame >= size_o[0]:
                    continue

                # Prepate input for next dimmension
                conv_input = input.view(b, c_i, size_i[0], -1)
                conv_input = conv_input[:, :, j, :].view((b, c_i) + size_i[1:])

                # Convolve
                frame_conv = \
                    self.conv_layers[i](conv_input)

                if empty_frames[out_frame] is None:
                    frame_results[out_frame] = frame_conv
                    empty_frames[out_frame] = 1
                else:
                    frame_results[out_frame] += frame_conv

        result = torch.stack(frame_results, dim=2)

        if self.use_bias:
            resultShape = result.shape
            result = result.view(b, resultShape[1], -1)
            for k in range(self.out_channels):
                result[:, k, :] += self.bias[k]
            return result.view(resultShape)
        else:
            return result


def conv4d(input_channels, output_channels, kernel_size, stride=1, padding=0, is_transposed=False, bias=True,
           groups=1, kernel_initializer=lambda x: torch.nn.init.normal_(x, mean=0.0, std=0.1), padding_mode='zeros'):
    return ConvNd(input_channels, output_channels, 4, kernel_size,
                  stride=stride, padding=padding, use_bias=bias, is_transposed=is_transposed,
                  padding_mode=padding_mode, groups=groups,
                  kernel_initializer=kernel_initializer,
                  bias_initializer=lambda x: torch.nn.init.normal_(x, mean=0.0, std=0.001))


class Augmentator():
    def __init__(self, loader):
        self.loader = loader

    def rotate(self, x: torch.Tensor, k: int) -> torch.Tensor:
        res_x = torch.rot90(x, k, (2, 3))
        return torch.rot90(res_x, k, (4, 5))

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        res_x = torch.flip(x, (3, ))
        return torch.flip(res_x, (5, ))

    def augmention(self, x, need_concatenate):
        result = {'x': x, 'x3': self.rotate(x, k=3), 'x6': self.rotate(x, k=2), 'x9': self.rotate(x, k=1)}
        result['rev_x'] = self.reverse(result['x'])
        result['rev_x3'] = self.reverse(result['x3'])
        result['rev_x6'] = self.reverse(result['x6'])
        result['rev_x9'] = self.reverse(result['x9'])
        if need_concatenate:
            x = result.pop('x')
            for key in result.keys():
                x = torch.cat((x, result[key]), dim=0)
            return x

        return result

    def generator(self, batch_size: int = 16, need_concatenate: bool = True):
        # TODO: Добавить режим случайного выбора аугментаций?
        for filename, x, target in self.loader.generator(batch_size):
            # target = torch.cat([target for _ in range(8)], axis=0) Is wrong


            yield filename, self.augmention(x, need_concatenate), target


class History:
    def __init__(self, metrics: list, report_period: int):
        self.length = 0
        self.report_period = report_period
        self.history_of_metrics = {}
        self.metrics = metrics
        for metric in self.metrics:
            self.history_of_metrics[metric] = []

    def append(self, **kwargs):
        if self.length % self.report_period == 0:
            self.report()

        for metric in self.metrics:
            value = kwargs[metric]
            self.history_of_metrics[metric].append(value)

        self.length += 1

    def report(self):
        result = {}
        msg = ''
        for metric in self.metrics:
            last_history = self.history_of_metrics[metric]
            if self.length > self.report_period:
                last_history = last_history[-self.report_period:]

            last_history.sort()

            result[metric] = {'min': last_history[0],
                              'mean': sum(last_history) / len(last_history),
                              'max': last_history[-1]}

            msg += f'{metric}: {result[metric]}'

        print(msg)


def sub_mean_by_neighbors(x, i, j, k, l):
    value = 0
    denominator = 0
    for a in range(-1, 1 + 1):
        for b in range(-1, 1 + 1):
            for c in range(-1, 1 + 1):
                for d in range(-1, 1 + 1):
                    try:
                        part = x[i + a, j + b, k + c, d + l]
                        if part == 0:
                            continue
                        denominator += 1
                        value += part
                    except IndexError:
                        pass
    return value / denominator


def mean_by_neighbors(x):
    new_x = np.array(x[0, 0])
    matrix = MammographMatrix().matrix
    for i in range(18):
        for j in range(18):
            for k in range(18):
                for l in range(18):
                    if matrix[i, j] == -1 or matrix[k, l] == -1:
                        print(x[0, 0, i, j, k, l])
                        continue

                    new_x[i, j, k, l] = sub_mean_by_neighbors(x[0, 0], i, j, k, l)



def visualize(x):
    pass


class LiteMeas:
    def __init__(self, path, use_cache=True):
        self.path = path
        self.filename = self.path.split('/')[-1]
        self.use_cache = use_cache
        self.matrix = MammographMatrix()
        self.raw_data = None
        self.amplitudes_data = None

        self.red_green_mask = None

        self.load()
        self.save()

    def save(self):
        if not os.path.exists('cache'):
            os.mkdir('cache')

        with open(f'cache/{self.filename}.pickle', 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self):
        cache_path = f'cache/{self.filename}.pickle'

        if self.use_cache and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                try:
                    self.__dict__ = pickle.load(f)
                    print('Взято из кэша')
                except (pickle.UnpicklingError, EOFError):
                    os.remove(cache_path)
            return None

        if '.txt' in self.filename:
            self.raw_data = self.txt_file_to_x(self.path)
        else:
            self.raw_data = parse_mammograph_raw_data(read_from_file_binary(self.path))
        self.amplitudes_data = self.restore_amplitudes()

    def restore_amplitudes(self):
        sar = np.sort(self.raw_data, axis=4)

        mx = sar[:, :, :, :, -1]
        mn = sar[:, :, :, :, 0]
        return (mx - mn) / 2

    def raw_signal(self, x_listener, y_listener, x_generator, y_generator):
        return self.raw_data[x_listener, y_listener, x_generator, y_generator]

    def __str__(self):
        return self.path.split('/')[-1]

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
                    if self.matrix.matrix[i, j] != -1:
                        x[i, j] = lst[self.matrix.matrix[i, j]]
                        print(i, j, self.matrix.matrix[i, j], lst[self.matrix.matrix[i, j]])
        res = np.zeros((18, 18, 18, 18, 80))

        for i, j in self.matrix.coordinate_generator():
            for k, l in self.matrix.coordinate_generator():
                for t in range(80):
                    res[i, j, k, l, t] = x[i, j, k, l] * np.sin(2 * np.pi * 100_000 * (t / 1200_000))

        return res


class MammographMatrix:
    def __init__(self):
        self.matrix = np.zeros((18, 18), dtype=np.int32) - 1
        self.matrix_inverse = np.zeros((18, 18), dtype=np.int32) + 1
        gen = iter(range(256))

        for i in range(6, 18 - 6):
            self.matrix[0, i] = next(gen)

        for i in range(4, 18 - 4):
            self.matrix[1, i] = next(gen)

        for i in range(3, 18 - 3):
            self.matrix[2, i] = next(gen)

        for i in range(2, 18 - 2):
            self.matrix[3, i] = next(gen)

        for j in range(2):
            for i in range(1, 18 - 1):
                self.matrix[4 + j, i] = next(gen)

        for j in range(6):
            for i in range(18):
                self.matrix[6 + j, i] = next(gen)

        for j in range(2):
            for i in range(1, 18 - 1):
                self.matrix[12 + j, i] = next(gen)

        for i in range(2, 18 - 2):
            self.matrix[14, i] = next(gen)

        for i in range(3, 18 - 3):
            self.matrix[15, i] = next(gen)

        for i in range(4, 18 - 4):
            self.matrix[16, i] = next(gen)

        for i in range(6, 18 - 6):
            self.matrix[17, i] = next(gen)

        for i in range(18):
            for j in range(18):
                if self.matrix[i, j] != -1:
                    self.matrix_inverse[i, j] = 0

    def coordinate_generator(self):
        for x in range(18):
            for y in range(18):
                if self.matrix[x, y] == -1:
                    continue
                yield x, y