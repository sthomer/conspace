import numpy as np
import json
from scipy.linalg import norm
import soundfile as sf
import os
from skimage import util
from scipy.fftpack import fft


def json_sequential(dimensions, res, suffix=''):
    lengths = [np.cumsum([length for segment in dimension.relative_lengths
                          for length in segment]) for dimension in dimensions]
    categories = [np.array([category for segment in dimension.segments
                            for category in segment]) for dimension in dimensions]
    lengths[2] = [lengths[1][index - 1] for index in lengths[2]]
    lengths[3] = [lengths[2][index - 1] for index in lengths[3]]
    if len(lengths) == 5:
        lengths[4] = [lengths[3][index - 1] for index in lengths[4]]
    pairs = [np.stack([lengths[i], categories[i]], axis=-1)
             for i in range(len(dimensions))]
    sequence = [[{'x0': pairs[d][i - 1 if i > 0 else 0][0],
                  'x': pairs[d][i][0],
                  'y': pairs[d][i][1]}
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


def json_spectrum(filename, res, suffix=''):
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


def json_annotation(filename, res, suffix='', word=True):
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


def json_semantic(dimensions, res, suffix=''):
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


# Only print top-level every 4 samples
def json_similarity(dimensions, res, suffix=''):
    # Spread segments like json_sequential
    lengths = [np.cumsum([length for segment in dimension.relative_lengths
                          for length in segment]) for dimension in dimensions]
    categories = [np.array([category for segment in dimension.segments
                            for category in segment]) for dimension in dimensions]
    lengths[2] = [lengths[1][index - 1] for index in lengths[2]]
    lengths[3] = [lengths[2][index - 1] for index in lengths[3]]
    pairs = [np.stack([lengths[i], categories[i]], axis=-1)
             for i in range(len(dimensions))]
    sequences = [[{'start': pairs[d][i - 1 if i > 0 else 0][0],
                   'end': pairs[d][i][0],
                   'category': pairs[d][i][1]}
                  for i in range(pairs[d].shape[0])]
                 for d in range(len(dimensions))]
    for d in range(len(dimensions)):
        if not sequences[d]:
            continue
        sequences[d][0]['start'] = 0

    # Split into length 1 elements
    flow = []
    for segment in sequences[-1]:
        for i in range(int(segment['start']), int(segment['end'])):
            flow.append(segment['category'])

    # Compute distances
    d = dimensions[-1]
    similarity = {}
    for x in d.stats:
        similarity[x] = {}
        for y in d.stats:
            similarity[x][y] = np.log(norm(d.stats[x].m_p - d.stats[y].m_p) + 1)

    # Heatmap like like json_spectrum
    output = []
    for a in range(0, len(flow), res):
        for b in range(0, len(flow), res):
            element = {'x': a, 'y': b,
                       'color': similarity[flow[a]][flow[b]]}
            output.append(element)

    with open('similarity.json', 'w') as file:
        json.dump(output, file)
    return output


def json_annotation(filename, res, suffix='', word=True):
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


# Only print top-level every 4 samples
def json_confusion(filename, dimensions, res, suffix=''):

    # Get phoneme annotation boundaries
    directory = os.path.dirname(filename)
    base = os.path.basename(filename).split('.')[0]
    load = directory + '/' + str(base) + '.PHN'
    with open(load, 'r') as file:
        ground = []
        for line in file:
            cols = line.split()
            ground.append({'start': int(cols[0]) / res,
                           'end': int(cols[1]) / res,
                           'label': cols[2]})
    # Split into length 1 elements
    phonemes = []
    for segment in ground:
        for i in range(int(segment['start']), int(segment['end'])):
            phonemes.append(segment['label'])

    # Spread segments like json_sequential
    lengths = [np.cumsum([length for segment in dimension.relative_lengths
                          for length in segment]) for dimension in dimensions]
    categories = [np.array([category for segment in dimension.segments
                            for category in segment]) for dimension in dimensions]
    lengths[2] = [lengths[1][index - 1] for index in lengths[2]]
    lengths[3] = [lengths[2][index - 1] for index in lengths[3]]
    pairs = [np.stack([lengths[i], categories[i]], axis=-1)
             for i in range(len(dimensions))]
    sequences = [[{'start': pairs[d][i - 1 if i > 0 else 0][0],
                   'end': pairs[d][i][0],
                   'category': pairs[d][i][1]}
                  for i in range(pairs[d].shape[0])]
                 for d in range(len(dimensions))]
    for d in range(len(dimensions)):
        if not sequences[d]:
            continue
        sequences[d][0]['start'] = 0

    # Split into length 1 elements
    flow = []
    for segment in sequences[-1]:
        for i in range(int(segment['start']), int(segment['end'])):
            flow.append(segment['category'])

    # Compute ln(x+1) distances
    d = dimensions[-1]
    similarity = {}
    max = 0
    for x in d.stats:
        similarity[x] = {}
        for y in d.stats:
            distance = np.log(norm(d.stats[x].m_p - d.stats[y].m_p) + 1)
            similarity[x][y] = distance
            max = max if max > distance else distance

    # Normalize to [0,1] with perfect similarity = 1 and difference = 0
    for row in similarity:
        for col in similarity[row]:
            el = similarity[row][col]
            el /= max
            similarity[row][col] = (-1 * el) + 1

    # Heatmap like like json_spectrum
    output = []
    for a in range(0, len(flow), res):
        for b in range(0, len(flow), res):
            match = 1 if phonemes[a] == phonemes[b] else 0
            distance = similarity[flow[a]][flow[b]] - match

            element = {'x': a, 'y': b,
                       'color': distance}
            output.append(element)

    with open('confusion.json', 'w') as file:
        json.dump(output, file)
    return output
