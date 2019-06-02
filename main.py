# import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.linalg import norm
from scipy.stats import entropy
# from sklearn import linear_model
from sklearn import gaussian_process
from skimage import util
import uuid
from collections import defaultdict, namedtuple
import warnings
import time
import soundfile as sf
import os
import functools
from multiprocessing.dummy import Pool
from itertools import repeat
import json

warnings.filterwarnings("ignore")  # annoying DivideByZero warning from entropy

Stats = namedtuple('Stats', 'm_s v_s m_p v_p')

timing = defaultdict(float)

pool = Pool()


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        timing[func.__name__] += run_time
        # print(f"{func.__name__!r}: {run_time:.4f}")
        return value

    return wrapper_timer


class Dimension:
    def __init__(self, level, radius, resolution):
        # Parameters
        self.level, self.radius, self.resolution = level, radius, resolution
        # Data structures
        self.stats = {}  # label -> Stats(m_s, v_s, m_p, v_p)
        self.unigram = defaultdict(int)
        self.bigram = defaultdict(dict)
        self.total = 0
        # Accumulators
        self.prev = None
        self.ongoing, self.lengths, self.current = [], [], []
        # Bookkeeping
        self.segments, self.relative_lengths = [], []

    def categorize_old(self, x):
        candidates = []
        for label in self.unigram:
            if label is None:
                continue
            centroid = self.stats[label].m_p
            distance = norm(centroid - x)
            radius = norm(np.sqrt(self.stats[label].v_p) * 3)  # i.e. 99.7%
            if distance <= radius:
                info = -np.log(self.unigram[label] / self.total)
                candidates.append((label, info))
        if not candidates:
            best = uuid.uuid1().hex
        else:
            best = max(candidates, key=lambda c: c[1])[0]
        return best

    def is_candidate(self, x, label):
        if label is not None:
            centroid = self.stats[label].m_p
            distance = norm(centroid - x)
            radius = norm(np.sqrt(self.stats[label].v_p) * 3)
            if distance <= radius:
                return label, -np.log(self.unigram[label] / self.total)

    def categorize(self, x):
        candidates = pool.starmap(
            self.is_candidate, zip(repeat(x), self.unigram.keys()))
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            best = uuid.uuid1().hex
        else:
            best = max(candidates, key=lambda c: c[1])[0]
        return best

    def update(self, curr, x, l):
        self.current.append(curr)
        self.ongoing.append(x)
        self.lengths.append(l)
        self.total += 1
        self.unigram[curr] += 1
        if curr not in self.bigram[self.prev]:
            self.bigram[self.prev][curr] = 0
        self.bigram[self.prev][curr] += 1

        if curr in self.stats:
            m_s, v_s, m_p, v_p = self.stats[curr]
        else:
            m_s = x
            v_s = np.ones(x.shape)
            m_p = np.zeros(x.shape)
            fill = (self.radius / 3) ** 2 / np.prod(x.shape)
            v_p = np.full(x.shape, fill)
        n_s = self.unigram[curr]
        M_s = m_s + (x - m_s) / n_s
        V_s = v_s + ((x - M_s) * (x - m_s) - v_s) / n_s if n_s > 1 else v_s
        M_p = (V_s * m_p + v_p * x) / (v_p + V_s)
        V_p = V_s * v_p / (V_s + v_p) if n_s > 1 else v_p
        self.stats[curr] = Stats(M_s.astype(np.complex64),  # save some mem
                                 V_s.astype(np.complex64),
                                 M_p.astype(np.complex64),
                                 V_p.astype(np.complex64))

    def segment(self, curr):
        entropy_prev = entropy(np.array(list(self.bigram[self.prev].values())))
        entropy_curr = entropy(np.array(list(self.bigram[curr].values())))
        info_prev = -np.log(self.unigram[self.prev] / self.total)
        info_curr = -np.log(self.unigram[curr] / self.total)
        return entropy_curr > entropy_prev or info_curr > info_prev

    def interpolate(self, segment):
        length, form = segment.shape[0], segment.shape[1:]
        segment_flat = segment.reshape(length, np.prod(form))
        regressor = gaussian_process.GaussianProcessRegressor()
        ticks = np.cumsum(self.lengths)
        ticks_present = ((ticks / ticks[-1]) * self.resolution)[:, np.newaxis]
        regressor.fit(ticks_present, segment_flat.view(np.float64))
        ticks_virtual = np.arange(self.resolution)[:, np.newaxis]
        trajectory_flat = regressor.predict(ticks_virtual).view(np.complex128)
        trajectory = trajectory_flat.reshape((self.resolution,) + form)
        return trajectory

    def abstract(self):
        segment = np.stack(self.ongoing, axis=0)
        trajectory = self.interpolate(segment)
        abstraction = fft(trajectory, axis=0)

        self.segments.append(self.current)
        self.relative_lengths.append(self.lengths)
        self.ongoing, self.lengths, self.current = [], [], []
        return abstraction, segment.shape[0]

    def perceive(self, x, l=1):
        curr = self.categorize(x)
        self.update(curr, x, l)
        abstraction, length = None, None
        if self.level < 3 and self.segment(curr):  # Too much mem beyond 3
            abstraction, length = self.abstract()
        self.prev = curr
        return abstraction, length


def json_sequential(dimensions):
    lengths = [np.cumsum([length for segment in dimension.relative_lengths
                          for length in segment]) for dimension in dimensions]
    categories = [np.array([category for segment in dimension.segments
                            for category in segment]) for dimension in dimensions]
    lengths[2] = [lengths[1][index] for index in lengths[2]]
    lengths[3] = [lengths[2][index] for index in lengths[3]]
    pairs = [np.stack([lengths[i], categories[i]], axis=-1)
             for i in range(dimensions.shape[0])]
    sequence = [[{'x0': pairs[d][i - 1 if i > 0 else 0][0],
                  'x': pairs[d][i][0],
                  'y': pairs[d][i][1]}
                 for i in range(pairs[d].shape[0])]
                for d in range(dimensions.shape[0])]
    for d in range(dimensions.shape[0]):
        sequence[d][0]['x0'] = 0

    categories = [set(category) for category in categories]
    orders = []
    for d in range(dimensions.shape[0]):
        order = [dimensions[d].segments[0][0]]
        while len(order) != len(categories[d]):
            successors = dimensions[d].bigram[order[-1]].items()
            successors = [s for s in successors if s[0] not in order]
            if successors:
                category = max(successors, key=lambda x: x[1])[0]
            else:
                category = list(set(categories[d]) - set(order))[0]
            order.append(category)
        orders.append(order)

    result = [{'categories': list(orders[i]), 'sequence': sequence[i]}
              for i in range(dimensions.shape[0])]
    with open('dimensions.json', 'w') as file:
        json.dump(result, file)
    return result


def json_spectrum(filename):
    res = 32
    data, samplerate = sf.read(filename)
    slices_time = util.view_as_windows(data, window_shape=(res,), step=res)
    slices = fft(slices_time)
    output = []
    for x in range(slices.shape[0]):
        for y in range(slices.shape[1] // 2):
            output.append({'x': x,
                           'y': y,
                           'color': np.abs(slices[x][y])})
    # out_name = 'spectrum-' + str(os.path.basename(filename)).split('.')[0]
    with open('spectrum.json', 'w') as file:
        json.dump(output, file)
    return output


def json_annotation(filename, word=True):
    res = 32
    directory = os.path.dirname(filename)
    base = os.path.basename(filename).split('.')[0]
    load = directory + '/' + str(base) + '.WRD' if word else '.PHN'

    with open(load, 'r') as file:
        output = []
        for line in file:
            cols = line.split()
            output.append({'x0': int(cols[0]) / res,
                           'x': int(cols[1]) / res,
                           'y': 0,
                           'label': cols[2]})
    # out_name = 'annotation-' + str(os.path.basename(filename)).split('.')[0]
    with open('wrd' if word else 'phn' + '-annotations.json', 'w') as file:
        json.dump(output, file)
    return output


def main(load=None, save=None, checkpoint=None, init=None, json=False, res=32):
    if not init:
        dimensions = [
            Dimension(0, 1e0, res),
            Dimension(1, 1e1, res),
            Dimension(2, 1e2, res),
            Dimension(3, 1e3, res),
        ]
    else:
        print('Loading dimensions')
        dimensions = np.load(init, allow_pickle=True)

    def perceive(x, l=1, i=0):
        superior, length = dimensions[i].perceive(x, l)
        if superior is not None:
            perceive(superior, length, i + 1)

    directory = os.path.dirname(load)
    if os.path.isfile(load) and load.split('.')[-1] == 'WAV':
        clips = [os.path.basename(load)]
    elif os.path.isdir(load):
        clips = [f for f in os.listdir(load) if f.split('.')[-1] == 'WAV']
    else:
        raise FileNotFoundError

    for clip in clips:

        print('Loading:', clip)
        data, sample_rate = sf.read(directory + '/' + clip)
        print('Duration:', len(data) / sample_rate, 's')

        slices_time = util.view_as_windows(data, window_shape=(res,), step=res)
        slices_freq = fft(slices_time)

        count = 0
        total = len(slices_freq)
        print('Perceiving:', total, 'moments')
        one_percent = total // (100 - 5)
        start = time.perf_counter()
        for signal in slices_freq:
            if count % one_percent == 0:
                print('Progress: ', count // one_percent, '%', sep='', end='\r')
            count += 1
            perceive(signal)
        print('Time:', f'{(time.perf_counter() - start):.4f}s')

        for d in dimensions:
            d.ongoing, d.lengths, d.current, d.prev = [], [], [], None

        if checkpoint:
            print('Checkpointing...')
            np.save(checkpoint, dimensions)
            print('Done')
    if save:
        print('Saving...')
        np.save(save, dimensions)
        print('Done')

    if json:
        json_sequential(dimensions)
        if os.path.isfile(load):
            json_spectrum(load)
            json_annotation(load, word=True)
            json_annotation(load, word=False)

    return dimensions


if __name__ == '__main__':
    directory = 'TIMIT/TRAIN/DR1/FCJF0/'
    dims = main(load=directory, save='FCJF0')
    # pass
