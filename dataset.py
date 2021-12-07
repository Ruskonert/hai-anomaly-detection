import numpy as np
from datetime import timedelta
import dateutil

from tqdm import trange, tqdm
from TaPR_pkg import etapr

import matplotlib.pyplot as plt


TIMESTAMP_FIELD = 'timestamp'
ATTACK_FIELD = 'attack'



"""

"""
def normalize(df, tag_min, tag_max):
    ndf = df.copy()
    for c in df.columns:
        if tag_min[c] == tag_max[c]:
            ndf[c] = df[c] - tag_min[c]
        else:
            ndf[c] = (df[c] - tag_min[c]) / (tag_max[c] - tag_min[c])
    return ndf


"""
"""
def get_best_threshold(xs, raw_ts, ts, att, t=None):
    result = {
        'best': 0.000,
        'best_tapr': None,
        'best_threshold': 0.000,
        'results': None
        }

    tapr_list = []

    if t is None:
        t = np.arange(0.0001, 0.0025, 0.0001)

    for THRESHOLD in tqdm(t):
        labels = put_labels(xs, THRESHOLD)
        final_labels = fill_blank(ts, labels, raw_ts)
        tapr_e = etapr.evaluate(anomalies=att, predictions=final_labels)
        if result['best'] < tapr_e['f1']:
            result['best'] = tapr_e['f1']
            result['best_threshold'] = THRESHOLD
            result['best_tapr'] = tapr_e
        tapr_list.append(tapr_e)
    
    result['results'] = tapr_list
    return result



"""

"""
def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs



"""

"""
def check_graph(xs, att, piece=2, THRESHOLD=None):
    l = xs.shape[0]
    chunk = l // piece
    fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))

    for i in range(piece):
        L = i * chunk
        R = min(L + chunk, l)
        xticks = range(L, R)
        axs[i].plot(xticks, xs[L:R])

        if len(xs[L:R]) > 0:
            peak = max(xs[L:R])
            axs[i].plot(xticks, att[L:R] * peak * 0.3)
            
        if THRESHOLD!=None:
            axs[i].axhline(y=THRESHOLD, color='r')

    plt.show()



"""

"""
def fill_blank(check_ts, labels, total_ts):
    def ts_generator():
        for t in total_ts:
            yield dateutil.parser.parse(t)

    def label_generator():
        for t, label in zip(check_ts, labels):
            yield dateutil.parser.parse(t), label

    g_ts = ts_generator()
    g_label = label_generator()
    final_labels = []

    try:
        current = next(g_ts)
        ts_label, label = next(g_label)
        
        while True:
            if current > ts_label:
                ts_label, label = next(g_label)
                continue
            elif current < ts_label:
                final_labels.append(0)
                current = next(g_ts)
                continue

            final_labels.append(label)
            current = next(g_ts)
            ts_label, label = next(g_label)

    except StopIteration:
        return np.array(final_labels, dtype=np.int8)



"""

"""
def boundary_check(df):
    x = np.array(df, dtype=np.float32)
    return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))



"""

"""
class TrainDataset():
    def __init__(self, train_raw):
        self.raw = train_raw
        self.ts = self.raw[TIMESTAMP_FIELD]
        self.col = self.raw.columns.drop([TIMESTAMP_FIELD])
        self.df = self.raw.drop([TIMESTAMP_FIELD], axis=1)
        

"""

"""
class ValidDataset():
    def __init__(self, train_raw):
        self.raw = train_raw
        self.ts = self.raw[TIMESTAMP_FIELD]
        self.att = self.raw[ATTACK_FIELD]
        self.col = self.raw.columns.drop([TIMESTAMP_FIELD, ATTACK_FIELD])
        self.df = self.raw.drop([TIMESTAMP_FIELD, ATTACK_FIELD], axis=1)


"""

"""
class HaiDataset():
    WINDOW_GIVEN = 59
    WINDOW_SIZE = 60
    def __init__(self, timestamps, df, stride=1, attacks=None, given=WINDOW_GIVEN, size=WINDOW_SIZE):
        self.ts = np.array(timestamps)
        self.windows_given = given
        self.windows_size = size
        self.tag_values = np.array(df, dtype=np.float32)
        self.valid_idxs = []

        for L in trange(len(self.ts) - size + 1):
            R = L + size - 1
            if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(self.ts[L]) == timedelta(seconds=size - 1):
                self.valid_idxs.append(L)

        valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(valid_idxs)
        
        print("# of valid windows:", self.n_idxs)

        if attacks is not None:
            self.attacks = np.array(attacks, dtype=np.float32)
            self.with_attack = True
        else:
            self.attacks = []
            self.with_attack = False
    
    def get_attack_set(self):
        attack_list = []
        if self.with_attack:
            for i in self.valid_idxs:
                last = i + self.windows_size - 1
                attack_list.append(self.attacks[last])
            return np.array(attack_list)
        else:
            return np.array([])

    def get_ts_set(self):
        timestamp = []
        for i in self.valid_idxs:
            last = i + self.windows_size - 1
            timestamp.append(self.ts[last])
        return np.array(timestamp)

    def get_train_set(self):
        X = []
        for i in self.valid_idxs:
            X.append(self.tag_values[i : i + self.windows_given])        
        return np.array(X)

    def get_label_set(self):
        y = []
        for i in self.valid_idxs:
            last = i + self.windows_size - 1
            y.append(self.tag_values[last])
        return np.array(y)