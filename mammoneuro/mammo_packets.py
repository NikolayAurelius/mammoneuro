# version 0.5.0

import struct
import numpy as np
from dahuffman import HuffmanCodec

HEADER_SIZE = 22
ACTUAL_HEADER_SIZE = 22
NUM_OF_CONTACTS = 256
POS_NUM_OF_CHARS = 3
POS_ALPHABET = 5
SINE_T_SIZE = 3


# size of data (3 bytes) | num of chars (u16) | (alphabet frequency (u8)+(3 bytes))*(num of chars)

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


mammograph_matrix = MammographMatrix().matrix

SINE_LOOKUP_TABLE_SIZE = 240
sine_lookup_table = ( \
    128, 131, 135, 138, 141, 145, 148, 151, \
    155, 158, 161, 164, 168, 171, 174, 177, \
    180, 183, 186, 189, 192, 195, 198, 200, \
    203, 206, 209, 211, 214, 216, 219, 221, \
    223, 225, 227, 230, 232, 233, 235, 237, \
    239, 240, 242, 244, 245, 246, 247, 249, \
    250, 251, 252, 252, 253, 254, 254, 255, \
    255, 256, 256, 256, 256, 256, 256, 256, \
    255, 255, 254, 254, 253, 252, 252, 251, \
    250, 249, 247, 246, 245, 244, 242, 240, \
    239, 237, 235, 233, 232, 230, 227, 225, \
    223, 221, 219, 216, 214, 211, 209, 206, \
    203, 200, 198, 195, 192, 189, 186, 183, \
    180, 177, 174, 171, 168, 164, 161, 158, \
    155, 151, 148, 145, 141, 138, 135, 131, \
    128, 125, 121, 118, 115, 111, 108, 105, \
    101, 98, 95, 92, 88, 85, 82, 79, \
    76, 73, 70, 67, 64, 61, 58, 56, \
    53, 50, 47, 45, 42, 40, 37, 35, \
    33, 31, 29, 26, 24, 23, 21, 19, \
    17, 16, 14, 12, 11, 10, 9, 7, \
    6, 5, 4, 4, 3, 2, 2, 1, \
    1, 0, 0, 0, 0, 0, 0, 0, \
    1, 1, 2, 2, 3, 4, 4, 5, \
    6, 7, 9, 10, 11, 12, 14, 16, \
    17, 19, 21, 23, 24, 26, 29, 31, \
    33, 35, 37, 40, 42, 45, 47, 50, \
    53, 56, 58, 61, 64, 67, 70, 73, \
    76, 79, 82, 85, 88, 92, 95, 98, \
    101, 105, 108, 111, 115, 118, 121, 125)

mammo_matrix_table = [[0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], \
                      [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], \
                      [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13],
                      [2, 14], \
                      [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [3, 12],
                      [3, 13], [3, 14], [3, 15], \
                      [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [4, 11], [4, 12],
                      [4, 13], [4, 14], [4, 15], [4, 16], \
                      [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [5, 11], [5, 12],
                      [5, 13], [5, 14], [5, 15], [5, 16], \
                      [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [6, 10], [6, 11],
                      [6, 12], [6, 13], [6, 14], [6, 15], [6, 16], [6, 17], \
                      [7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10], [7, 11],
                      [7, 12], [7, 13], [7, 14], [7, 15], [7, 16], [7, 17], \
                      [8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10], [8, 11],
                      [8, 12], [8, 13], [8, 14], [8, 15], [8, 16], [8, 17], \
                      [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], [9, 10], [9, 11],
                      [9, 12], [9, 13], [9, 14], [9, 15], [9, 16], [9, 17], \
                      [10, 0], [10, 1], [10, 2], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9],
                      [10, 10], [10, 11], [10, 12], [10, 13], [10, 14], [10, 15], [10, 16], [10, 17], \
                      [11, 0], [11, 1], [11, 2], [11, 3], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [11, 9],
                      [11, 10], [11, 11], [11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17], \
                      [12, 1], [12, 2], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9], [12, 10],
                      [12, 11], [12, 12], [12, 13], [12, 14], [12, 15], [12, 16], \
                      [13, 1], [13, 2], [13, 3], [13, 4], [13, 5], [13, 6], [13, 7], [13, 8], [13, 9], [13, 10],
                      [13, 11], [13, 12], [13, 13], [13, 14], [13, 15], [13, 16], \
                      [14, 2], [14, 3], [14, 4], [14, 5], [14, 6], [14, 7], [14, 8], [14, 9], [14, 10], [14, 11],
                      [14, 12], [14, 13], [14, 14], [14, 15], \
                      [15, 3], [15, 4], [15, 5], [15, 6], [15, 7], [15, 8], [15, 9], [15, 10], [15, 11], [15, 12],
                      [15, 13], [15, 14], \
                      [16, 4], [16, 5], [16, 6], [16, 7], [16, 8], [16, 9], [16, 10], [16, 11], [16, 12], [16, 13], \
                      [17, 6], [17, 7], [17, 8], [17, 9], [17, 10], [17, 11]]

mammo_matrix_table2 = [None, None, None, None, None, None, 0, 1, 2, 3, 4, 5, None, None, None, None, None, None, \
                       None, None, None, None, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, None, None, None, None, \
                       None, None, None, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, None, None, None, \
                       None, None, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, None, None, \
                       None, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, None, \
                       None, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, None, \
                       74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, \
                       92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, \
                       110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, \
                       128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, \
                       146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, \
                       164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, \
                       None, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, None, \
                       None, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, None, \
                       None, None, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, None, None, \
                       None, None, None, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, None, None, None, \
                       None, None, None, None, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, None, None, None, None, \
                       None, None, None, None, None, None, 250, 251, 252, 253, 254, 255, None, None, None, None, None,
                       None]


# u8 phase;
# u8 amp;
# u8 dc;

class Sine:
    def __init__(self, phase, amp, dc):
        self.phase = phase
        self.amp = amp
        self.dc = dc


def cos(x):
    assert isinstance(x, int)
    x += SINE_LOOKUP_TABLE_SIZE // 4
    x = x % SINE_LOOKUP_TABLE_SIZE
    if (x < 0):
        x += SINE_LOOKUP_TABLE_SIZE
    return sine_lookup_table[x]


def matrix_rotate_90(m):
    m_zipped = list(zip(*reversed(m)))
    rotated = []
    for l in m_zipped:
        rotated.append(list(l))
    return rotated


def two_list_in_one(X1, X2):
    X = []
    # print("len(X1) = ", len(X1), "len(X2) = ", len(X2))
    for i in range(len(X1)):
        x = X1[i]
        x.extend(X2[i])
        X.append(x)

    return X


def sine_add(in_buff, period, sine):
    out_buff = []
    freq_num = SINE_LOOKUP_TABLE_SIZE  # numerator
    freq_den = period  # denominator
    for i in range(len(in_buff)):
        y_cos = cos(int((i - sine.phase) * freq_num // freq_den))
        y_cos1 = y_cos
        y_cos = int(int(y_cos * sine.amp) >> 8)  # normalize to amp of input sine
        y_cos2 = y_cos
        out_buff.append(((in_buff[i]) + y_cos + sine.dc) % 256)  # remove sine and dc component

    return out_buff


def decode(data):
    # decode binary data with special header for decompress algorithm
    real_size = int.from_bytes(data[:POS_NUM_OF_CHARS], byteorder='little')
    num_of_chars = int.from_bytes(data[POS_NUM_OF_CHARS:POS_NUM_OF_CHARS + 2], byteorder='big')
    # int(data[POS_NUM_OF_CHARS])

    dummy_list = []

    print("num_of_chars ", num_of_chars)
    for i in range(num_of_chars):
        c_ch = data[POS_ALPHABET + i * 4]

        c_ch_freq = int.from_bytes(data[POS_ALPHABET + 1 + i * 4:POS_ALPHABET + 1 + i * 4 + 3], byteorder='big')

        dummy_list.append((c_ch, c_ch_freq))

    alphabet_frequencies = dict(dummy_list)
    print("alphabet_frequencies", alphabet_frequencies)
    codec = HuffmanCodec.from_frequencies(alphabet_frequencies)
    codec.print_code_table()
    return codec.decode(data[POS_ALPHABET + num_of_chars * 4:])


def parse_compressed_mammograph_packets(data):
    # return frames, number of samples in one mesure and period of sinus in counts of ADC \
    # frames stored in binary format
    header = list(struct.unpack('HHB', data[:HEADER_SIZE]))
    print("header ", header)

    period = header[0]
    N_samples = header[2]

    _data = data[HEADER_SIZE:]

    frames = []
    for i in range(1):
        # i=0
        # get size
        compressed_sine_t_size = int.from_bytes(_data[:2], byteorder='little')
        print("compressed_sine_t_size ", compressed_sine_t_size)
        _data = _data[2:]
        # get sine data
        sine_data_params = decode(_data[:compressed_sine_t_size])
        _data = _data[compressed_sine_t_size:]
        print("decoded sine_data_params size ", len(sine_data_params))
        # get size compressed high bytes
        compressed_sine_high_bytes_size = int.from_bytes(_data[:2], byteorder='little')
        _data = _data[2:]
        print("compressed_sine_high_bytes_size ", compressed_sine_high_bytes_size)
        # get high bytes
        sine_high_bytes_data = decode(_data[:compressed_sine_high_bytes_size])
        _data = _data[compressed_sine_high_bytes_size:]
        print("decoded sine_high_bytes_data size ", len(sine_high_bytes_data))
        # get low bytes
        sine_low_bytes_data = _data[:N_samples * NUM_OF_CONTACTS]

        _data = _data[N_samples * NUM_OF_CONTACTS:]
        frame = [sine_data_params, sine_high_bytes_data, sine_low_bytes_data]
        frames.append(frame)

    return frames, N_samples, period


def matrix256_to_18x18(matrix):
    out_matrix = [[0] * 18 for i in range(18)]
    for i in range(256):
        x = mammo_matrix_table[i][0]
        y = mammo_matrix_table[i][1]
        out_matrix[x][y] = matrix[i]
    return out_matrix


def parse_mammograph_raw_data(data):
    header = list(struct.unpack('11H', data[:ACTUAL_HEADER_SIZE]))
    print("header ", header)

    header_version = header[0]
    mammo_data_size = header[1]

    period = header[8]
    freq = header[9]
    N_samples = header[10]

    print("period: ", period, " freq: ", freq, " N_samples: ", N_samples)
    N_point_samples = N_samples * 256 * 256

    _data = data[HEADER_SIZE:HEADER_SIZE + N_point_samples * 2]
    print(len(_data))

    # > for big endian
    # h for short signed integer

    # print (_data)
    # shape (256*256*N_samples)
    list_of_int = list(struct.unpack('>' + str(N_point_samples) + 'h', _data))
    # list_of_int = list(struct.unpack('>16h', _data))
    # print (list_of_int )
    frame_sampled_size = 4 * 256 * N_samples
    list_of_int_combined_samples = []
    # make samples combined by point measure
    # s1,s1,s1,s1,s2,s2,s2,s2,s3,s3,s3,s3,s4,s4,s4,s4...
    # s1,s2,s3,s4...s1,s2,s3,s4...s1,s2,s3,s4...s1,s2,s3,s4...
    print("length list_of_int ", len(list_of_int))
    # print("list_of_int_combined_samples ", list_of_int_combined_samples)
    # for i in range(int(256*256/4)):
    for i in range(int(256 / 4)):
        # print (len (ch1))
        # shape (256*N_samples),(256*N_samples),(256*N_samples),(256*N_samples)
        ch1 = list_of_int[0 + i * frame_sampled_size: 0 + (i + 1) * frame_sampled_size: 4]
        ch2 = list_of_int[1 + i * frame_sampled_size: 1 + (i + 1) * frame_sampled_size: 4]
        ch3 = list_of_int[2 + i * frame_sampled_size: 2 + (i + 1) * frame_sampled_size: 4]
        ch4 = list_of_int[3 + i * frame_sampled_size: 3 + (i + 1) * frame_sampled_size: 4]

        # shape (256,80),(256,80),(256,80),(256,80)
        ch1 = [ch1[i:i + N_samples] for i in range(0, len(ch1), N_samples)]
        ch2 = [ch2[i:i + N_samples] for i in range(0, len(ch2), N_samples)]
        ch3 = [ch3[i:i + N_samples] for i in range(0, len(ch3), N_samples)]
        ch4 = [ch4[i:i + N_samples] for i in range(0, len(ch4), N_samples)]

        # print (ar.shape)
        # print (len(ch1), len(ch2), len(ch3), len(ch4))
        # list_of_int_combined_samples.append(ch1)
        # list_of_int_combined_samples.append(ch2)
        # list_of_int_combined_samples.append(ch3)
        # list_of_int_combined_samples.append(ch4)
        # shape (64,4,256,80)
        list_of_int_combined_samples.append([ch1, ch2, ch3, ch4])

        # print(ch1)
        # print(ch2)
        # print(ch3)
        # print(ch4)

    # print("length list_of_int_combined_samples ", len(list_of_int_combined_samples) )
    # ar = np.array(list_of_int_combined_samples)
    # print (ar.shape)
    # shape (80)
    x = [0 for x in range(N_samples)]
    # shape (18,18,80)
    dummy_adc_frames = [[x] * 18 for i in range(18)]
    # shape (18,18,18,18,80)
    adc_frames = [[dummy_adc_frames] * 18 for i in range(18)]

    # shape (18,18,80)
    dummy_dac_frames = [[x] * 18 for i in range(18)]
    # shape (18,18,18,18,80)
    dac_frames = [[dummy_dac_frames] * 18 for i in range(18)]

    # ar = np.array(adc_frames)
    # print (ar.shape)

    print("------_adc_frame_quadrant-------")
    _adc_frame_quadrant = [[], [], [], []]
    adc_frame_quadrant = [[], [], [], []]
    for i in range(len(list_of_int_combined_samples)):
        # shape (4,256,80)
        _a = list_of_int_combined_samples[i]

        # shape (256,80)
        _adc_frame_quadrant[0] = _a[0]
        _adc_frame_quadrant[1] = _a[1]
        _adc_frame_quadrant[2] = _a[2]
        _adc_frame_quadrant[3] = _a[3]

        # location of qudrants
        # 0 3
        # 1 2

        for j in range(4):
            x = [0 for x in range(N_samples)]
            dummy_dac_frames = [[x] * 18 for i in range(18)]
            for k in range(256):
                x = mammo_matrix_table[k][0]
                y = mammo_matrix_table[k][1]
                # shape (18,18,80)
                dummy_dac_frames[x][y] = _adc_frame_quadrant[j][k]
            # shape (4,64,18,18,80)
            adc_frame_quadrant[j].append(dummy_dac_frames)
    ar = np.array(adc_frame_quadrant)
    print(ar.shape)
    # print ("len(adc_frame_quadrant[0])", len(adc_frame_quadrant[0]))
    # print ("len(adc_frame_quadrant[1])", len(adc_frame_quadrant[1]))
    # print ("len(adc_frame_quadrant[2])", len(adc_frame_quadrant[2]))
    # print ("len(adc_frame_quadrant[3])", len(adc_frame_quadrant[3]))    
    # for j in range(4):
    # adc_frame_quadrant[j]=adc_frame_quadrant[j][0]

    _adc_frame_quadrant = adc_frame_quadrant
    adc_frame_quadrant = [[], [], [], []]

    for j in range(4):
        x = [0 for x in range(N_samples)]
        # shape (18,18,80)
        dummy_adc_frames = [[x] * 18 for i in range(18)]
        du_adc = dummy_adc_frames
        # a = [0,0,0,0,0,0]
        a = [du_adc, du_adc, du_adc, du_adc, du_adc, du_adc]

        a.extend(_adc_frame_quadrant[j][0:3])
        adc_frame_quadrant[j].append(a)

        a = [du_adc, du_adc, du_adc, du_adc]
        a.extend(_adc_frame_quadrant[j][3:8])
        adc_frame_quadrant[j].append(a)
        a = [du_adc, du_adc, du_adc]
        a.extend(_adc_frame_quadrant[j][8:14])
        adc_frame_quadrant[j].append(a)
        a = [du_adc, du_adc]
        a.extend(_adc_frame_quadrant[j][14:21])
        adc_frame_quadrant[j].append(a)
        a = [du_adc]
        a.extend(_adc_frame_quadrant[j][21:29])
        adc_frame_quadrant[j].append(a)
        a = [du_adc]
        a.extend(_adc_frame_quadrant[j][29:37])
        adc_frame_quadrant[j].append(a)
        adc_frame_quadrant[j].append((_adc_frame_quadrant[j][37:46]))
        adc_frame_quadrant[j].append((_adc_frame_quadrant[j][46:55]))
        adc_frame_quadrant[j].append((_adc_frame_quadrant[j][55:64]))

        ar = np.array(adc_frame_quadrant[j])
        # print("adc_frame_quadrant[j] ",adc_frame_quadrant[j])
        # print("len(adc_frame_quadrant[j]) ",len(adc_frame_quadrant[j]))
        print(ar.shape)

        # print (len(adc_frame_quadrant[j]))
        # print("--------------------------------------------------------------------")
        # print(adc_frame_quadrant[j])

    # rotate matrices
    adc_frame_quadrant[3] = matrix_rotate_90(adc_frame_quadrant[3])

    adc_frame_quadrant[2] = matrix_rotate_90(adc_frame_quadrant[2])
    adc_frame_quadrant[2] = matrix_rotate_90(adc_frame_quadrant[2])

    adc_frame_quadrant[1] = matrix_rotate_90(adc_frame_quadrant[1])
    adc_frame_quadrant[1] = matrix_rotate_90(adc_frame_quadrant[1])
    adc_frame_quadrant[1] = matrix_rotate_90(adc_frame_quadrant[1])

    # print(adc_frame_quadrant[1])
    adc_frame = []
    # print (adc_frame_quadrant[0],adc_frame_quadrant[3])

    # print (np.array(adc_frame_quadrant[0]).shape)
    # print (np.array(adc_frame_quadrant[3]).shape)
    adc_frame = two_list_in_one(adc_frame_quadrant[0], adc_frame_quadrant[3])
    adc_frame.extend(two_list_in_one(adc_frame_quadrant[1], adc_frame_quadrant[2]))

    ar = np.array(adc_frame)

    print(ar.shape)

    return np.array(adc_frame)
    # return adc_frames


def parse_uncompressed_mammograph_packets(data):
    header = list(struct.unpack('HHB', data[:ACTUAL_HEADER_SIZE]))
    print("header ", header)

    period = header[0]
    N_samples = header[2]

    _data = data[HEADER_SIZE:]

    N_point_samples = N_samples * 256 * 256

    _data = data[HEADER_SIZE:HEADER_SIZE + N_point_samples * 2]

    # > for big endian
    # h for short signed integer

    # print (_data)
    list_of_int = list(struct.unpack(str(N_point_samples) + 'h', _data))

    # shape (256*256,80)
    list_sample_points = [list_of_int[i:i + N_samples] for i in range(0, len(list_of_int), N_samples)]

    print("list_sample_points", len(list_sample_points))
    # shape (64,1024,80)
    list_frames = [list_sample_points[i:i + 1024] for i in range(0, len(list_sample_points), 1024)]
    print("len list_frames: ", len(list_frames))
    # dummy_sample_point

    # shape (80)
    x = [0 for x in range(N_samples)]
    # shape (18,18,80)
    dummy_adc_frames = [[x] * 18 for i in range(18)]
    # shape (18,18,18,18,80)
    adc_frames = [[dummy_adc_frames] * 18 for i in range(18)]

    dummy_dac_frames = [[x] * 18 for i in range(18)]
    dac_frames = [[dummy_dac_frames] * 18 for i in range(18)]

    ar = np.array(adc_frames)
    # print (ar.shape)

    adc_frame_quadrant = [[], [], [], []]
    for i in range(len(list_frames)):
        _adc_frame_quadrant = [[], [], [], []]

        # shape (1024,80)
        list_frame = list_frames[i]
        # print ("len _adc_frame_quadrant: ", len(_adc_frame_quadrant) )
        # shape (4,256,80)
        _adc_frame_quadrant[0] = list_frame[0::4]
        _adc_frame_quadrant[1] = list_frame[1::4]
        _adc_frame_quadrant[2] = list_frame[2::4]
        _adc_frame_quadrant[3] = list_frame[3::4]

        # print ("len _adc_frame_quadrant: ", len(_adc_frame_quadrant[0]) )
        for j in range(4):
            # shape (N_samples)
            x = [0 for x in range(N_samples)]
            # shape (18,18,N_samples)
            dummy_dac_frames = [[x] * 18 for i in range(18)]
            for k in range(256):
                x = mammo_matrix_table[k][0]
                y = mammo_matrix_table[k][1]
                dummy_dac_frames[x][y] = _adc_frame_quadrant[j][k]
            adc_frame_quadrant[j].append(dummy_dac_frames)
        # print ("len(_adc_frame_quadrant[0])", len(_adc_frame_quadrant[0]))
        # print ("len(_adc_frame_quadrant[1])", len(_adc_frame_quadrant[1]))
        # print ("len(_adc_frame_quadrant[2])", len(_adc_frame_quadrant[2]))
        # print ("len(_adc_frame_quadrant[3])", len(_adc_frame_quadrant[3]))

    # print ("len(adc_frame_quadrant[0][0])", len(adc_frame_quadrant[0][0]))
    # print ("len(adc_frame_quadrant[1][0])", len(adc_frame_quadrant[1][0]))
    # print ("len(adc_frame_quadrant[2][0])", len(adc_frame_quadrant[2][0]))
    # print ("len(adc_frame_quadrant[3][0])", len(adc_frame_quadrant[3][0]))        

    _adc_frame_quadrant = adc_frame_quadrant
    adc_frame_quadrant = [[], [], [], []]

    # print ("_adc_frame_quadrant[0]", _adc_frame_quadrant[0])    
    # print ("len(_adc_frame_quadrant[0])", len(_adc_frame_quadrant[0]))
    # print ("len(_adc_frame_quadrant[1])", len(_adc_frame_quadrant[1]))
    # print ("len(_adc_frame_quadrant[2])", len(_adc_frame_quadrant[2]))
    # print ("len(_adc_frame_quadrant[3])", len(_adc_frame_quadrant[3]))

    # print(_adc_frame_quadrant[0])
    for j in range(4):
        x = [0 for x in range(N_samples)]
        # shape (18,18,80)
        dummy_adc_frames = [[x] * 18 for i in range(18)]
        du_adc = dummy_adc_frames
        # a = [0,0,0,0,0,0]
        a = [du_adc, du_adc, du_adc, du_adc, du_adc, du_adc]
        a.extend(_adc_frame_quadrant[j][0:3])
        adc_frame_quadrant[j].append(a)
        a = [du_adc, du_adc, du_adc, du_adc]
        a.extend(_adc_frame_quadrant[j][3:8])
        adc_frame_quadrant[j].append(a)
        a = [du_adc, du_adc, du_adc]
        a.extend(_adc_frame_quadrant[j][8:14])
        adc_frame_quadrant[j].append(a)
        a = [du_adc, du_adc]
        a.extend(_adc_frame_quadrant[j][14:21])
        adc_frame_quadrant[j].append(a)
        a = [du_adc]
        a.extend(_adc_frame_quadrant[j][21:29])
        adc_frame_quadrant[j].append(a)
        a = [du_adc]
        a.extend(_adc_frame_quadrant[j][29:37])
        adc_frame_quadrant[j].append(a)
        adc_frame_quadrant[j].append((_adc_frame_quadrant[j][37:46]))
        adc_frame_quadrant[j].append((_adc_frame_quadrant[j][46:55]))
        adc_frame_quadrant[j].append((_adc_frame_quadrant[j][55:64]))

    adc_frame_quadrant[3] = matrix_rotate_90(adc_frame_quadrant[3])

    adc_frame_quadrant[2] = matrix_rotate_90(adc_frame_quadrant[2])
    adc_frame_quadrant[2] = matrix_rotate_90(adc_frame_quadrant[2])

    adc_frame_quadrant[1] = matrix_rotate_90(adc_frame_quadrant[1])
    adc_frame_quadrant[1] = matrix_rotate_90(adc_frame_quadrant[1])
    adc_frame_quadrant[1] = matrix_rotate_90(adc_frame_quadrant[1])

    adc_frames = []
    adc_frames = two_list_in_one(adc_frame_quadrant[0], adc_frame_quadrant[3])
    adc_frames.extend(two_list_in_one(adc_frame_quadrant[1], adc_frame_quadrant[2]))

    # x = mammo_matrix_table[i][0]
    # y = mammo_matrix_table[i][1]
    # adc_frames[x][y] = adc_frame

    return np.array(adc_frames)


def parse_frame(frame, N_samples, period):
    # return frame values as list of lists with integer values \
    # [[x1,x2,x3...xn],[y1,y2,y3,...yn],...[k1,k2,k3...,kn]]
    bb = bytearray(b'')
    values = []
    sine_data_params = frame[0]
    sine_high_bytes_data = frame[1]
    sine_low_bytes_data = frame[2]
    frame_values = []
    for i in range(NUM_OF_CONTACTS):
        sine = Sine(sine_data_params[i * 3], sine_data_params[1 + i * 3], sine_data_params[2 + i * 3])
        out_buff = sine_add(frame[1][i * N_samples:(i + 1) * N_samples], period, sine)
        _ch = []
        values = []
        # make i16 variables
        for j in range(N_samples):
            v = int(frame[2][j + i * N_samples])
            v += (out_buff[j] << 8)
            values.append(v)

        frame_values.append(values)
    return frame_values


def save_to_binary_file(file_name, data):
    file = open(file_name, 'wb')
    file.write(data)
    file.close()


def save_to_file(file_name, data):
    file = open(file_name, 'w')
    file.write(data)
    file.close()


def read_from_file_binary(file_name):
    file = open(file_name, 'rb')
    data = file.read()
    file.close()
    return data
