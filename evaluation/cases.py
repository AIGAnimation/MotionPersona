"""Test cases: one case = (persona, body, style, seed), and the test-set expansions."""
import csv
from dataclasses import dataclass
from pathlib import Path

from data.loco_dataset import STYLE_VOCAB, SUBJECT_VOCAB

STYLES = tuple(s for s in STYLE_VOCAB if s != 't_pose')          # the nine captured styles
SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True, order=True)
class Case:
    persona: str
    body: int
    style: str
    seed: int

    @property
    def key(self):
        return f'{self.persona}__{self.body:03d}__{self.style}__s{self.seed}'

    @property
    def stem(self):
        return f'{self.persona}__{self.body:03d}__{self.style}'


def load_bodies(bodies_csv='data/bodies/bodies.csv'):
    rows = {int(r['id']): r for r in csv.DictReader(open(bodies_csv))}
    return rows


def own_body(persona, bodies):
    for b, r in bodies.items():
        if r['kind'] == 'real' and r['name'] == persona:
            return b
    raise KeyError(f'no captured body named {persona!r}')


def split_files(split_dir):
    d = Path(split_dir)
    out = {}
    for name in ('heldout_bodies', 'sweep_bodies', 'withheld_cells', 'heldout_takes'):
        p = d / f'{name}.csv'
        if p.exists():
            out[name] = [r for r in csv.DictReader(l for l in open(p) if not l.startswith('#'))]
    return out


def build_cases(test_set, *, bodies, split_dir=None, seeds=SEEDS, personas=SUBJECT_VOCAB, sweep_persona='p02'):
    """Expand a named test set into cases."""
    personas = list(personas)
    if test_set == 'own':
        return [Case(p, own_body(p, bodies), s, k) for p in personas for s in STYLES for k in seeds]
    if test_set == 'own_neutral':                             # tab:guidance / step-response: every performer, neutral, own body
        return [Case(p, own_body(p, bodies), 'neutral', k) for p in personas for k in seeds]
    if test_set.startswith('own1:'):                           # 'own1:<persona>' -- one performer, all styles (dev / demo)
        p = test_set.split(':', 1)[1]
        return [Case(p, own_body(p, bodies), s, k) for s in STYLES for k in seeds]
    sf = split_files(split_dir) if split_dir else {}
    if test_set in ('seen', 'withheld', 'heldout', 'sweep_shape', 'sweep_style', 'sweep_persona'):
        if not sf:
            raise SystemExit(f'test set {test_set!r} needs the split files (data/eval/split_v2)')
    if test_set == 'heldout':
        hb = [int(r['body']) for r in sf['heldout_bodies']]
        return [Case(p, b, 'neutral', k) for p in personas for b in hb for k in seeds]
    if test_set in ('seen', 'withheld'):
        train_sweep = [int(r['body']) for r in sf['sweep_bodies'] if r['role'] == 'train']
        withheld = {(r['persona'], int(r['body'])) for r in sf['withheld_cells']}
        out = []
        for p in personas:
            for b in train_sweep:
                if ((p, b) in withheld) == (test_set == 'withheld'):
                    out += [Case(p, b, 'neutral', k) for k in seeds]
        return out
    if test_set == 'sweep_shape':
        sb = [int(r['body']) for r in sf['sweep_bodies']]
        return [Case(sweep_persona, b, 'neutral', k) for b in sb for k in seeds]
    if test_set == 'sweep_style':
        sb = [int(r['body']) for r in sf['sweep_bodies']]
        return [Case(sweep_persona, b, s, k) for s in STYLES for b in sb for k in seeds]
    if test_set == 'sweep_persona':                            # paper 6.2: all 44 identities on each sweep body, neutral
        sb = [int(r['body']) for r in sf['sweep_bodies']]
        return [Case(p, b, 'neutral', k) for p in personas for b in sb for k in seeds]
    raise SystemExit(f'unknown test set {test_set!r}')


def cases_from_csv(path):
    out = []
    for r in csv.DictReader(open(path)):
        out.append(Case(r['persona'], int(r['body']), r['style'], int(r['seed'])))
    return out
