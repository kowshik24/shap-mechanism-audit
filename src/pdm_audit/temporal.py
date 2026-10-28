"""Past-only context and whole-cycle chronological split helpers."""
import numpy as np

CONTEXT = ['Previous wear', 'Previous torque', 'Previous temperature difference',
           'Cycle sample age', 'Past mean torque']


def add_context(df):
    out = df.copy()
    cycle = (out['Tool wear [min]'].diff() < 0).cumsum().to_numpy()
    out['Wear cycle'] = cycle
    grouped = out.groupby('Wear cycle', sort=False)
    out['Previous wear'] = grouped['Tool wear [min]'].shift(1).fillna(0)
    out['Previous torque'] = grouped['Torque [Nm]'].shift(1).fillna(out['Torque [Nm]'])
    out['Previous temperature difference'] = grouped['Temp_diff [K]'].shift(1).fillna(out['Temp_diff [K]'])
    out['Cycle sample age'] = grouped.cumcount()
    out['Past mean torque'] = grouped['Torque [Nm]'].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(out['Torque [Nm]'])
    return out


def chronological_split(df):
    groups = np.array(sorted(df['Wear cycle'].unique()))
    first, second = int(.6*len(groups)), int(.8*len(groups))
    return tuple(df.index[df['Wear cycle'].isin(ids)].to_numpy()
                 for ids in [groups[:first], groups[first:second], groups[second:]])


def inner_folds(df, training_rows):
    subset = df.loc[training_rows]
    groups = np.array(sorted(subset['Wear cycle'].unique()))
    for end, stop in [(.3, .5), (.5, .7), (.7, 1.)]:
        a, b = int(end*len(groups)), int(stop*len(groups))
        yield (subset.index[subset['Wear cycle'].isin(groups[:a])].to_numpy(),
               subset.index[subset['Wear cycle'].isin(groups[a:b])].to_numpy())
