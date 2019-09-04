from skimage import util
import soundfile as sf
import os

from dimension import *
from generate_json import *


def main(load=None, save=None, checkpoint=None, init=None, json=False,
         res=16, scale=1):
    if not init:
        dimensions = [
            Dimension(0, scale * 1e0, res),
            Dimension(1, scale * 1e1, res),
            Dimension(2, scale * 1e2, res),
            Dimension(3, scale * 1e3, res),
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
        # data, sample_rate = sf.read(directory + '/' + clip)
        data, sample_rate = sf.read(clip)
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
        json_sequential(dimensions, res, json)
        json_semantic(dimensions, res, json)
        # json_similarity(dimensions, 8, json)
        if os.path.isfile(load):
            json_spectrum(load, res, json)
            # json_annotation(load, res, json, word=True)
            # json_annotation(load, res, json, word=False)

    return dimensions


if __name__ == '__main__':
    directory = 'arpeggio-AC.WAV'
    mem = 'arpeggio-AC.npy'
    main(load=directory,
         save=None,
         init=None,
         json='-arpeggio-AC',
         scale=1000,
         res=256
         )
