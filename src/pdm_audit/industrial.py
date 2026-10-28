"""Measured-system metadata, structural controls, and reference-capacity checks."""
from html.parser import HTMLParser
from math import ceil
import re
from urllib.parse import urljoin
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


class SourceTable(HTMLParser):
    """Read table row metadata rather than infer operating load from file names."""
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell, self.link = [], None, None, None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row = []
        elif tag in ('td', 'th') and self.row is not None:
            self.cell = {'text': '', 'links': []}
        elif tag == 'a' and self.cell is not None:
            self.link = {'href': dict(attrs).get('href', ''), 'text': ''}

    def handle_data(self, data):
        if self.cell is not None:
            self.cell['text'] += data
        if self.link is not None:
            self.link['text'] += data

    def handle_endtag(self, tag):
        if tag == 'a' and self.link is not None:
            self.cell['links'].append(self.link)
            self.link = None
        elif tag in ('td', 'th') and self.cell is not None:
            self.row.append(self.cell)
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def table_metadata(html, source_url):
    parser = SourceTable()
    parser.feed(html)
    records, diameter = {}, None
    for row in parser.rows:
        if len(row) < 6 or not any(c['links'] for c in row[3:]):
            continue
        try:
            load = int(row[1]['text'].strip())
            rpm = float(row[2]['text'].strip())
        except ValueError:
            continue
        text = row[0]['text'].strip().replace('"', '')
        if text:
            diameter = float(text)
        for column, fault in [(3, 'IR'), (4, 'B'), (5, 'OR')]:
            for link in row[column]['links']:
                label = link['text'].strip()
                if not re.fullmatch(r'(?:IR|B|OR)(?:007|014|021)(?:@6)?_[0-3]', label):
                    continue
                if fault == 'OR' and '@6' not in label:
                    continue
                assert label not in records
                records[label] = {'load_hp': load, 'diameter_in': diameter,
                                  'fault': fault, 'approximate_rpm': rpm,
                                  'outer_race_position': '6:00' if fault == 'OR' else None,
                                  'url': urljoin(source_url, link['href'])}
    return records


def split_support(estimator, columns, allowed):
    used = sorted({columns[int(j)] for tree in estimator.estimators_
                   for j in tree.tree_.feature if j >= 0})
    outside = sorted(set(used)-set(allowed))
    return {'used_features': used, 'outside_anchor_splits': outside,
            'coordinate_invariant': not outside,
            'trees_checked': len(estimator.estimators_)}


def recording_binary(metadata, probabilities, fault):
    votes = pd.Series(np.asarray(probabilities), index=metadata.recording.to_numpy()).groupby(level=0).mean()
    truth = metadata.groupby('recording').fault.first().loc[votes.index] == fault
    pred = votes >= .5
    tp, tn = int((truth & pred).sum()), int((~truth & ~pred).sum())
    return {'F1': float(f1_score(truth, pred, zero_division=0)),
            'sensitivity': float(tp/truth.sum()), 'specificity': float(tn/(~truth).sum()),
            'tp': tp, 'tn': tn, 'positive_n': int(truth.sum()), 'negative_n': int((~truth).sum()),
            'recordings': [{'id': k, 'truth': bool(truth[k]), 'probability': float(votes[k]),
                            'prediction': bool(pred[k])} for k in votes.index]}


def rank_capacity(batches, alpha=.05):
    rank = ceil((batches+1)*(1-alpha))
    return {'calibration_batches': batches, 'alpha': alpha, 'rank': rank,
            'finite_threshold_available': rank <= batches,
            'minimum_tail_resolution': 1/(batches+1),
            'threshold_if_unavailable': None if rank <= batches else 'infinity; abstain'}
