"""Learned condition tokens shared by the latent prior (network/latfm.py) and the raw DDPM baseline (network/models.py).

Style: one embedding row per captured style label.
Persona (paper, section 5): a persona is a closed-set performer ID plus three typed attributes,
so it enters as up to four learned tokens instead of the CLIP feature of a prompt.  Both are condition tokens that
are NEVER masked -- `model.cond_mask_prob` drops the history only.
"""
import torch.nn as nn

from data.loco_dataset import ATTR_VOCABS, STYLE_VOCAB, SUBJECT_VOCAB

STYLE_CONDS = ('none', 'embed')
PERSONA_CONDS = ('text', 'id', 'attr', 'id_attr', 'none')
PERSONA_LEARNED = ('id', 'attr', 'id_attr')


class StyleEmbed(nn.Module):
    """`model.style_cond: embed`: the style enters as its own learned token (nn.Embedding over
    data/loco_dataset.py STYLE_VOCAB, from meta label.style) instead of a clause inside the text prompt."""

    def __init__(self, latent_dim, n_styles=None):
        super().__init__()
        # n_styles: a dataset-provided vocabulary (meta.pkl style_vocab, gestures); None = STYLE_VOCAB (locomotion)
        self.table = nn.Embedding(len(STYLE_VOCAB) if n_styles is None else int(n_styles), latent_dim)

    def forward(self, style_idx):
        assert style_idx is not None, 'style_cond=embed needs style_idx (LocoDataset meta labels)'
        return self.table(style_idx.long()).unsqueeze(0)             # [1, B, d] like the other condition tokens


class PersonaEmbed(nn.Module):
    """`model.persona_cond` in (id, attr, id_attr): the persona as learned tokens.  id = nn.Embedding over
    data/loco_dataset.py SUBJECT_VOCAB (44 performers, from manifest subject / raw BVH directory); attr = three
    tables over ROLE / AFFILIATION / DOMINANCE_VOCAB (typed: 'moderate' affiliation and 'moderate' dominance are
    different rows), from meta label.  A batch row outside the vocabulary (-1) fails loudly instead of indexing a
    wrong row."""

    def __init__(self, latent_dim, kind, n_subjects=None):
        super().__init__()
        assert kind in PERSONA_LEARNED, kind
        self.use_id, self.use_attr = kind in ('id', 'id_attr'), kind in ('attr', 'id_attr')
        if self.use_id:
            # n_subjects: a dataset-provided vocabulary (meta.pkl subject_vocab, gestures); None = SUBJECT_VOCAB
            self.id_table = nn.Embedding(len(SUBJECT_VOCAB) if n_subjects is None else int(n_subjects), latent_dim)
        if self.use_attr:
            self.attr_tables = nn.ModuleList([nn.Embedding(len(v), latent_dim) for v in ATTR_VOCABS])

    @property
    def n_tokens(self):
        return int(self.use_id) + 3 * int(self.use_attr)

    def forward(self, subject_idx, attr_idx):
        """subject_idx (B,) rows of SUBJECT_VOCAB, attr_idx (B, 3) rows of ATTR_VOCABS -> list of [1, B, d] tokens."""
        toks = []
        if self.use_id:
            assert subject_idx is not None and bool((subject_idx >= 0).all()), 'persona ID token: a batch row has no known subject'
            toks.append(self.id_table(subject_idx.long()).unsqueeze(0))
        if self.use_attr:
            assert attr_idx is not None and bool((attr_idx >= 0).all()), 'persona attribute tokens: a batch row has no label'
            toks += [table(attr_idx[:, j].long()).unsqueeze(0) for j, table in enumerate(self.attr_tables)]
        return toks


def persona_tokens(model, text_feat, persona):
    """The persona part of a token sequence: the CLIP text token of a prompt (persona_cond=text), the learned
    ID / attribute tokens, or nothing (persona_cond=none)."""
    if getattr(model, 'use_text', True):
        assert text_feat is not None, 'persona_cond=text needs text_feat'
        return [model.text_embed(text_feat)]
    if getattr(model, 'persona', None) is not None:
        assert persona is not None, f'persona_cond={model.persona_cond} needs (subject_idx, attr_idx)'
        return model.persona(*persona)
    return []


def cond_token_kwargs(model, cond):
    """The learned-token conditions a model wants from a batch's `conditions` dict; empty for a text-prompt model.
    Missing rows fail loudly (a silently dropped condition would train wrong)."""
    kw = {}
    if getattr(model, 'use_style', False):
        if 'style_idx' not in cond:
            raise SystemExit('model.style_cond=embed needs style_idx in the batch (LocoDataset clips carry it from meta label.style)')
        kw['style_idx'] = cond['style_idx']
    if getattr(model, 'persona', None) is not None:
        need = (['subject_idx'] if model.persona.use_id else []) + (['attr_idx'] if model.persona.use_attr else [])
        missing = [k for k in need if k not in cond]
        if missing:
            raise SystemExit(f'model.persona_cond={model.persona_cond} needs {missing} in the batch '
                             '(LocoDataset clips carry them from manifest subject / meta label)')
        kw['persona'] = (cond.get('subject_idx'), cond.get('attr_idx'))
    return kw


def vocab_sizes(meta):
    """(n_subjects, n_styles) for the learned token tables: None, None (= the fixed locomotion vocabularies) unless the
    dataset meta carries its own `subject_vocab` / `style_vocab` (speech-driven gestures).  A locomotion checkpoint
    therefore builds exactly the tables it was trained with and loads strict."""
    meta = meta or {}
    sv, st = meta.get('subject_vocab'), meta.get('style_vocab')
    return (len(sv) if sv else None), (len(st) if st else None)
