"""Nested train-only views; each subset balances the two elevation rings."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ViewSetting:
    key: str
    count: int
    layout: str
    names: tuple[str, ...]


def settings(train_names):
    pool=set(train_names)
    expected={f'view_{i:03d}.png' for i in range(48) if i%4!=0}
    if pool!=expected:
        raise ValueError('Source split is outside the approved two-ring48-view policy')
    uniform={6:[1,9,17],12:list(range(1,24,4)),24:list(range(1,24,2))}
    available=[i for i in range(24) if i%4!=0]
    result=[]
    def names(indices):
        return tuple(f'view_{i+offset:03d}.png' for offset in [0,24] for i in indices)
    for n in [6,12,24]:
        result.append(ViewSetting(f'uniform_{n:02d}',n,'uniform_orbit',names(uniform[n])))
    result.append(ViewSetting('all_36',36,'shared_full_orbit',tuple(sorted(pool))))
    for n in [6,12,24]:
        result.append(ViewSetting(f'arc_{n:02d}',n,'contiguous_arc',names(available[:n//2])))
    for row in result:
        assert len(row.names)==len(set(row.names))==row.count and set(row.names)<=pool
    for prefix in ['uniform','arc']:
        sets=[set(next(r.names for r in result if r.key==f'{prefix}_{n:02d}')) for n in [6,12,24]]+[pool]
        assert all(a<b for a,b in zip(sets,sets[1:]))
    return result
