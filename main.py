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
        self.unigram = {}  # defaultdict(int)
        self.bigram = {}  # defaultdict(dict)
        self.total = 0
        # Accumulators
        self.prev = None
        self.ongoing, self.lengths, self.current = [], [], []
        # Bookkeeping
        self.segments, self.relative_lengths = [], []

    # Either results in everything categorized together,
    #   or nothing categorized together, depending on prior radius
    def categorize_union(self, x):
        candidates = []
        for label in self.unigram:
            if label is None:
                continue
            centroid = self.stats[label].m_p
            distance = norm(centroid - x)
            radius = norm(np.sqrt(self.stats[label].v_p) * 3)  # i.e. 99.7%
            if distance <= radius:
                candidates.append(label)

        category = uuid.uuid1().hex
        if candidates:
            # Combine stats
            m_p = np.mean([self.stats[label].m_p for label in candidates], axis=0)
            v_p = np.mean([self.stats[label].v_p for label in candidates], axis=0)
            n_s = np.sum([self.unigram[label] for label in candidates])
            m_s = np.sum([(self.unigram[label] / n_s) * self.stats[label].m_s
                          for label in candidates], axis=0)
            v_s = np.sum([(self.unigram[label] / n_s) * self.stats[label].v_s
                          for label in candidates], axis=0)
            self.stats[category] = Stats(m_s, v_s, m_p, v_p)
            for label in candidates:
                del self.stats[label]

            # Combine unigram
            self.unigram[category] = n_s
            for label in candidates:
                del self.unigram[label]

            # Combine bigram: first layer
            combined = {}
            for label in candidates:
                overlap = set(combined.keys()) & set(self.bigram[label].keys())
                for c in overlap:
                    combined[c] += self.bigram[label][c]
                    del self.bigram[label][c]
                combined = {**combined, **self.bigram[label]}
                del self.bigram[label]
            self.bigram[category] = combined

            # Combine bigram: second layer
            total = 0
            for c in self.bigram:
                if c is None:
                    continue
                for label in candidates:
                    if label in self.bigram[c]:
                        total += self.bigram[c][label]
            for c in self.bigram:
                if c is None:
                    continue
                for label in candidates:
                    if label in self.bigram[c]:
                        del self.bigram[c][label]
                        self.bigram[c][category] = total

            # Update labels in segments
            self.current = [category if c in candidates else c for c in self.current]
            self.segments = [[category if c in candidates else c for c in segment]
                             for segment in self.segments]

            # Remove combined candidates
            if self.prev in candidates:
                self.prev = category

        return category

    def categorize_single_thread(self, x):
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

        if self.prev not in self.unigram:
            self.unigram[self.prev] = 0
        if curr not in self.unigram:
            self.unigram[curr] = 0
        if self.prev not in self.bigram:
            self.bigram[self.prev] = {}
        if curr not in self.bigram:
            self.bigram[curr] = {}
        if curr not in self.bigram[self.prev]:
            self.bigram[self.prev][curr] = 0

        self.total += 1
        self.unigram[curr] += 1
        self.bigram[self.prev][curr] += 1

        if curr in self.stats:
            m_s, v_s, m_p, v_p = self.stats[curr]
        else:
            m_s = x
            v_s = np.zeros(x.shape)  # np.ones(x.shape) # motherfucker
            m_p = x  # np.zeros(x.shape) # doesn't matter if v_s == 0
            fill = (self.radius / 3) ** 2 / np.prod(x.shape)
            v_p = np.full(x.shape, fill)
        n_s = self.unigram[curr]
        M_s = m_s + (x - m_s) / n_s
        V_s = (v_s + ((x - M_s) * (x - m_s) - v_s) / n_s) if n_s > 1 else v_s
        M_p = (V_s * m_p + v_p * x) / (v_p + V_s)
        V_p = (V_s * v_p / (V_s + v_p)) if n_s > 1 else v_p
        c = Stats(M_s,  # .astype(np.complex64),  # save some mem
                  V_s,  # .astype(np.complex64),
                  M_p,  # .astype(np.complex64),
                  V_p)  # .astype(np.complex64))
        self.stats[curr] = c
        self.ongoing.append(c.m_p)  # abstraction over categories
        # self.ongoing.append(x)  # abstraction over specifics
        self.lengths.append(l)
        self.current.append(curr)

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


def json_sequential(dimensions, res):
    lengths = [np.cumsum([length for segment in dimension.relative_lengths
                          for length in segment]) for dimension in dimensions]
    categories = [np.array([category for segment in dimension.segments
                            for category in segment]) for dimension in dimensions]
    lengths[2] = [lengths[1][index - 1] for index in lengths[2]]
    lengths[3] = [lengths[2][index - 1] for index in lengths[3]]
    pairs = [np.stack([lengths[i], categories[i]], axis=-1)
             for i in range(len(dimensions))]
    sequence = [[{'x0': pairs[d][i - 1 if i > 0 else 0][0],
                  'x': pairs[d][i][0],
                  'y': pairs[d][i][1],
                  'stroke': 'blue' if i % 2 == 0 else 'orange'}
                 for i in range(pairs[d].shape[0])]
                for d in range(len(dimensions))]
    for d in range(len(dimensions)):
        if not sequence[d]:
            continue
        sequence[d][0]['x0'] = 0

    categories = [set(category) for category in categories]
    orders = []
    for d in range(len(dimensions)):
        if not sequence[d]:
            continue
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

    result = []
    for d in range(len(dimensions)):
        if not sequence[d]:
            continue
        result.append({'categories': list(orders[d]), 'sequence': sequence[d]})

    with open('dimensions.json', 'w') as file:
        json.dump(result, file)
    return result


def json_spectrum(filename, res):
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


def json_annotation(filename, res, word=True):
    directory = os.path.dirname(filename)
    base = os.path.basename(filename).split('.')[0]
    load = directory + '/' + str(base) + ('.WRD' if word else '.PHN')
    with open(load, 'r') as file:
        output = []
        for line in file:
            cols = line.split()
            output.append({'x0': int(cols[0]) / res,
                           'x': int(cols[1]) / res,
                           'y': 0,
                           'label': cols[2]})
    # out_name = 'annotation-' + str(os.path.basename(filename)).split('.')[0]
    with open(('wrd' if word else 'phn') + '-annotations.json', 'w') as file:
        json.dump(output, file)
    return output


def json_semantic(dimensions, res):
    output = []
    for dimension in dimensions:
        categories = []
        for category, (_, _, m_p, v_p) in dimension.stats.items():
            size = norm(m_p)
            radius = norm(np.sqrt(v_p) * 3)
            categories.append({'x': str(size), 'y': str(radius), 'size': 10})
        categories.sort(key=lambda r: float(r['x']))
        output.append(categories)
    with open('dimensions_semantic.json', 'w') as file:
        json.dump(output, file)
    return output


def main(load=None, save=None, checkpoint=None, init=None, json=False, res=16):
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

    clip_count = 0
    for clip in clips:
        clip_count += 1
        print('Loading:', clip, '(', clip_count, '/', len(clips), ')')
        data, sample_rate = sf.read(directory + '/' + clip)
        print('Duration:', len(data) / sample_rate, 's')

        slices_time = util.view_as_windows(data, window_shape=(res,), step=res)
        slices_freq = fft(slices_time)

        for d in dimensions:
            d.segments, d.relative_lengths = [], []
            d.ongoing, d.lengths, d.current, d.prev = [], [], [], None

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
        print('Time:', time.perf_counter() - start, 's')

        for d in dimensions:
            d.segments.append(d.current)
            d.relative_lengths.append(d.lengths)
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
        json_sequential(dimensions, res)
        json_semantic(dimensions, res)
        if os.path.isfile(load):
            json_spectrum(load, res)
            json_annotation(load, res, word=True)
            json_annotation(load, res, word=False)

    return dimensions


if __name__ == '__main__':
    directory = 'TIMIT/TRAIN/DR1/FCJF0/'
    mem = 'finish_line.npy'
    dims = main(load=directory,
                save=mem,
                init=None,
                json=False)
    single = main(load=directory + 'SA1.WAV',
                  save=None,
                  init=mem,
                  json=True)
