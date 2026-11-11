"""Optional Weights & Biases mirror of a training run's metrics.jsonl.

metrics.jsonl and result.json stay the source of truth; this sends the same records to W&B so runs can be
watched live and compared across the team. ``wandb`` is imported only when a run asks for it, so tests,
``--toy`` and machines without it never need the package.

One record per step, logged at W&B step = training step. Nested values are flattened to chartable keys:
the gradient probe to ``probe/per_group/head/cosine``, adaptive allocation's per-layer draw lists (method B)
to ``allocation/writers/0``. The run id is kept in ``<output>/wandb_id.txt`` so
``--resume`` continues the same W&B run. Steps rerun after a crash are dropped by W&B (it ignores steps it
has already seen); resume is exact (test_training), so the values it already holds are the same.

Use: ``python dust.py ... --wandb PROJECT [--wandb-entity TEAM] [--wandb-group NAME]``, after
``wandb login``. ``WANDB_MODE=offline`` records locally under <output>/wandb for a later ``wandb sync``.
"""
import secrets
from pathlib import Path

ID_FILE = 'wandb_id.txt'


def flatten(record, prefix=''):
    """{'probe': {'cosine': 0.3}, 'writers': [4, 2]} -> {'probe/cosine': 0.3, 'writers/0': 4, 'writers/1': 2}."""
    flat = {}
    for key, value in record.items():
        name = f'{prefix}{key}'
        if isinstance(value, (list, tuple)):
            value = dict(enumerate(value))
        if isinstance(value, dict):
            flat.update(flatten(value, name + '/'))
        else:
            flat[name] = value
    return flat


class WandbLogger:
    """Mirror of one run. ``config`` is the run's metadata (config.json); ``resume`` reuses the stored id."""

    def __init__(self, output, config=None, *, project, entity=None, group=None, name=None, tags=None,
                 mode=None, resume=False):
        import wandb  # optional dependency; only runs that log to W&B need it

        self.wandb = wandb
        output = Path(output)
        id_file = output / ID_FILE
        if resume and id_file.is_file():
            run_id = id_file.read_text().strip()
        else:
            run_id = secrets.token_hex(4)  # W&B ids: short, [a-z0-9]
            id_file.write_text(run_id + '\n')
        self.run = wandb.init(project=project, entity=entity, group=group, name=name or output.name,
                              tags=tags, config=config, id=run_id, resume='allow', dir=str(output), mode=mode)
        for key in ('flops', 'flops_executed', 'forward_equivalents', 'train_seconds', 'gpu_hours'):
            self.run.define_metric(key)  # usable as an x-axis: val loss vs FLOPs is the headline plot
        self.run.define_metric('val_loss', summary='min')

    def log(self, record):
        self.run.log(flatten(record), step=record['step'])

    def finish(self, result):
        summary = {k: v for k, v in result.items() if k != 'history'}
        self.run.summary.update(flatten(summary))
        self.run.finish()
