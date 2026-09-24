"""R45/V302 continuation with coverage-constrained training hard batches."""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

import train_v302_hard_age_depth_time_grl_long as training
from miwu.tidyvoice_w2vbert2_axial_head import ContinuedGRLW2VBert2
from train_v260_age_balanced_residual_redimnet2 import BalancedAgeComplementSampler


class CoverageHardSampler(BalancedAgeComplementSampler):
    """Cover eligible identities before reusing them, within the original groups."""

    def __init__(self, dataset, config, prototypes, young_groups, adult_indexes):
        super().__init__(dataset, config, prototypes, young_groups, adult_indexes)
        adult = set(adult_indexes)
        groups = {
            'preschool': list(dataset.child_group_indexes.values()),
            'school': list(young_groups),
            'ocean_adult': [[i for i in dataset.ocean_gender_indexes[g] if i in adult]
                            for g in ('m', 'f')],
            '3d': [dataset.domain_indexes['3d']],
            'stcmds': [dataset.domain_indexes['stcmds']],
            'cv': [dataset.domain_indexes['cv']],
        }
        self.allowed = {}
        self.macro_of = {}
        for macro, parts in groups.items():
            for part in parts:
                part = sorted(set(part))
                if len(part) < 4:
                    continue  # Preserve the baseline's eligibility, report exclusions.
                for index in part:
                    if index in self.macro_of:
                        raise ValueError('Overlapping demographic groups')
                    self.allowed[index] = part
                    self.macro_of[index] = macro
            assert sorted(i for i, m in self.macro_of.items() if m == macro) == self.anchors[macro]
        p = F.normalize(prototypes.detach().float().cpu(), dim=1)
        self.similarity = p @ p.T
        self.warmup_steps = 1 if self.steps == 2 else int(config["head_steps"])
        self.speakers = dataset.speakers
        self.excluded = [s for i, s in enumerate(self.speakers) if i not in self.macro_of]

    def __iter__(self):
        pending = {name: set(values) for name, values in self.anchors.items()}
        counts = Counter()
        wide = ('3d', 'ocean_adult', 'stcmds', 'cv')

        def cluster(name):
            if not pending[name]:
                pending[name] = set(self.anchors[name])
            anchor = random.choice(sorted(pending[name]))
            candidates = [i for i in self.allowed[anchor] if i != anchor]
            # Prefer uncovered identities; within each set retain acoustic hardness.
            candidates.sort(key=lambda i: (
                i not in pending[name],
                counts[i] if i not in pending[name] else 0,
                -float(self.similarity[anchor, i]), i))
            selected = [anchor] + candidates[:3]
            assert len(set(selected)) == 4
            pending[name].difference_update(selected)
            counts.update(selected)
            return selected

        for step in range(self.steps):
            if step == self.warmup_steps:
                # Classifier-only warmup must not consume model-training coverage.
                pending = {name: set(values) for name, values in self.anchors.items()}
                counts.clear()
            age = 'preschool' if step % 2 == 0 else 'school'
            indexes = cluster(age) + cluster(wide[(step // 2) % len(wide)])
            random.shuffle(indexes)
            duration = random.choices(self.durations, weights=self.duration_weights, k=1)[0]
            yield [(index, duration) for index in indexes]


class ObservedLoader:
    """Count batches actually consumed by optimization, excluding prefetch/preview."""

    def __init__(self, loader, head_steps):
        self.loader, self.head_steps = loader, head_steps
        self.counts = defaultdict(Counter)
        self.hardness = defaultdict(list)
        self.batches = 0

    def __iter__(self):
        sampler = self.loader.batch_sampler
        for batch in self.loader:
            phase = 'classifier_warmup' if self.batches < self.head_steps else 'model_training'
            labels = batch[-1].tolist()
            self.counts[phase].update(labels)
            by_macro = defaultdict(list)
            for index in labels:
                by_macro[sampler.macro_of[index]].append(index)
            for macro, group in by_macro.items():
                assert len(group) == 4 and len(set(group)) == 4
                for j, left in enumerate(group):
                    self.hardness[phase + '/' + macro].extend(
                        float(sampler.similarity[left, right]) for right in group[j + 1:])
            self.batches += 1
            yield batch

    def write_report(self, output):
        sampler = self.loader.batch_sampler
        phases = {}
        for phase, counts in self.counts.items():
            groups = {}
            for macro, indexes in sampler.anchors.items():
                values = [counts[i] for i in indexes]
                hardness = self.hardness[phase + '/' + macro]
                groups[macro] = {
                    'eligible_identities': len(indexes), 'slots': sum(values),
                    'unique_seen': sum(v > 0 for v in values),
                    'minimum_count': min(values), 'maximum_count': max(values),
                    'mean_negative_cosine': sum(hardness) / len(hardness) if hardness else None,
                }
            phases[phase] = groups
        report = {'actual_consumed_batches': self.batches, 'phases': phases,
                  'excluded_under_unchanged_baseline_group_rules': sampler.excluded,
                  'validation_audio_used': False,
                  'counts_by_training_identity': {
                      phase: {sampler.speakers[i]: c for i, c in sorted(counts.items())}
                      for phase, counts in self.counts.items()}}
        (output / 'observed_sampling.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'actual_sampling': phases, 'excluded': sampler.excluded}), flush=True)


original_train = training.train


def train_with_observation(model, head, loader, config, output, smoke=False):
    observed = ObservedLoader(loader, 1 if smoke else config['head_steps'])
    original_train(model, head, observed, config, output, smoke)
    observed.write_report(output)


def main():
    training.DepthTimeGRLW2VBert2 = ContinuedGRLW2VBert2
    training.BalancedAgeComplementSampler = CoverageHardSampler
    training.train = train_with_observation
    defaults = {'--config': 'configs/v302_hard_age_depth_time_grl_long.yaml',
                '--prototype-file': 'outputs/v328_corrected_prototypes.pt',
                '--output-dir': 'outputs/v358_coverage_grl'}
    for flag, value in defaults.items():
        if flag not in sys.argv:
            sys.argv.extend([flag, value])
    training.main()
    output = Path(sys.argv[sys.argv.index('--output-dir') + 1])
    path = output / 'metadata.json'
    metadata = json.loads(path.read_text())
    metadata.update(
        architecture='Unchanged R45/V302 complete depth-time speaker head',
        initialization=ContinuedGRLW2VBert2.initial_checkpoint,
        teacher='Frozen original V302',
        changed_inference_architecture=False,
        research_change='Coverage-constrained hard identity sampling only',
        baseline_semifinal_lb=93.35674,
        inference_runtime='Unchanged successful R45',
        validation_audio_used=False)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
