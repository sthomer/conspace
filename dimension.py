import numpy as np
from scipy.fftpack import fft
from scipy.linalg import norm
from sklearn import gaussian_process
import uuid
from collections import namedtuple
import warnings
from multiprocessing.dummy import Pool
from itertools import repeat

from benchmarking import *

warnings.filterwarnings("ignore")  # annoying DivideByZero warning from entropy
Stats = namedtuple('Stats', 'm_s v_s m_p v_p')
pool = Pool()


class Dimension:
    @timer
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

    @timer
    def categorize_loop(self, x):
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

    @timer
    def categorize(self, x):
        if len(self.unigram) == 0:
            return uuid.uuid1().hex
        x = x.ravel()
        labels, centroids, variances = self.vectors()
        candidates = self.norms(x, labels, centroids, variances)
        return candidates[0] if candidates.size else uuid.uuid1().hex

    @timer
    def vectors(self):
        # TODO: Slow (40s) These can be cached in self.stats
        # Let's hope they iterate the same way for each comprehension?
        labels = np.array([label for label in self.unigram if label is not None])
        centroids = np.array([self.stats[label].m_p.ravel()
                              for label in self.unigram if label is not None])
        variances = np.array([self.stats[label].v_p.ravel()
                              for label in self.unigram if label is not None])
        return labels, centroids, variances

    @timer
    def norms(self, x, labels, centroids, variances):
        # TODO: VERY slow (190s) Variances can be cached in self.stats
        radii = norm(np.sqrt(variances * 3), axis=1)
        distances = norm(centroids - x, axis=1)
        candidates = labels[np.nonzero(distances <= radii)]
        return candidates

    @timer
    def is_candidate(self, x, label):
        if label is not None:
            centroid = self.stats[label].m_p
            distance = norm(centroid - x)
            radius = norm(np.sqrt(self.stats[label].v_p) * 3)
            if distance <= radius:
                return label, -np.log(self.unigram[label] / self.total)

    @timer
    def categorize_multi_thread(self, x):
        candidates = pool.starmap(
            self.is_candidate, zip(repeat(x), self.unigram.keys()))
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            best = uuid.uuid1().hex
        else:
            best = max(candidates, key=lambda c: c[1])[0]
        return best

    @timer
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
            v_s = np.zeros(x.shape)
            m_p = x
            fill = (self.radius / 3) ** 2 / np.prod(x.shape)
            v_p = np.full(x.shape, fill)
        n_s = self.unigram[curr]
        M_s = m_s + (x - m_s) / n_s
        V_s = (v_s + ((x - M_s) * (x - m_s) - v_s) / n_s) if n_s > 1 else v_s
        if np.isclose(v_p + V_s, np.zeros(v_p.shape)).any():
            M_p, V_p = m_p, v_p
        else:
            M_p = (V_s * m_p + v_p * x) / (v_p + V_s)
            V_p = (V_s * v_p / (V_s + v_p)) if n_s > 1 else v_p
        c = Stats(M_s,
                  V_s,
                  M_p,
                  V_p)
        self.stats[curr] = c
        self.ongoing.append(c.m_p)  # abstraction over categories
        self.lengths.append(l)
        self.current.append(curr)

    @timer
    def segment(self, curr):
        # entropy_prev = entropy(np.array(list(self.bigram[self.prev].values())))
        # entropy_curr = entropy(np.array(list(self.bigram[curr].values())))
        info_prev = -np.log(self.unigram[self.prev] / self.total)
        info_curr = -np.log(self.unigram[curr] / self.total)
        # return entropy_curr > entropy_prev or info_curr > info_prev
        threshold = 0
        return info_curr - info_prev > threshold

    @timer
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

    @timer
    def abstract(self):
        segment = np.stack(self.ongoing, axis=0)
        trajectory = self.interpolate(segment)
        abstraction = fft(trajectory, axis=0)

        self.segments.append(self.current)
        self.relative_lengths.append(self.lengths)
        self.ongoing, self.lengths, self.current = [], [], []
        return abstraction, segment.shape[0]

    @timer
    def perceive(self, x, l=1):
        curr = self.categorize(x)
        self.update(curr, x, l)
        abstraction, length = None, None
        if self.level < 1 and self.segment(curr):  # Too much mem beyond 3
            abstraction, length = self.abstract()
        self.prev = curr
        return abstraction, length
