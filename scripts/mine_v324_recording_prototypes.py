#!/usr/bin/env python3
"""Extract two training recordings per identity with frozen V302; no training."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from miwu.tidyvoice_w2vbert2_depth_time_head import DepthTimeGRLW2VBert2
from miwu.encoder import load_audio
from train_v1 import crop_or_repeat, stable_split
from train_v45_layerwise_compensation import ShortFullPairDataset
from train_v55_four_domain_local_temporal import add_commonvoice
from train_v7_pyramid_large import add_stcmds
from train_v176_child_dual_axis_redimnet2 import add_childmandarin, rebuild_indexes


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def summary(values):
    values = values.detach().float().cpu()
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise RuntimeError('Empty or nonfinite training diagnostic')
    quantiles = (0., .05, .25, .5, .75, .95, 1.)
    return {
        'count': values.numel(), 'mean': values.mean().item(),
        'std': values.std(unbiased=False).item(),
        'quantiles': {str(q): torch.quantile(values, q).item() for q in quantiles},
    }


def class_margins(queries, prototypes, labels):
    similarities = queries @ prototypes.T
    rows = torch.arange(len(labels))
    own = similarities[rows, labels].clone()
    similarities[rows, labels] = -float('inf')
    other = similarities.max(dim=1).values
    return own, other, own - other


def margins_report(values):
    own, other, margin = values
    return {'own_class_cosine': summary(own),
            'strongest_other_class_cosine': summary(other),
            'own_class_margin': summary(margin),
            'positive_margin_fraction': (margin > 0).float().mean().item()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=Path('tmp/v263_r40_fresh'))
    parser.add_argument('--three-d-root', default='data/processed/3dspeaker')
    parser.add_argument('--ocean-root', default='data/processed/speechocean')
    parser.add_argument('--stcmds-root', default='data/processed/stcmds/ST-CMDS-20170001_1-OS')
    parser.add_argument('--commonvoice-root', default='data/processed/commonvoice17-train')
    parser.add_argument('--childmandarin-root', default='data/raw/childmandarin/train')
    parser.add_argument('--grl-checkpoint', default='outputs/v223_s2_grl_head/grl_head.ckpt')
    parser.add_argument('--checkpoint', default='outputs/v302_hard_age_depth_time_grl_long/final.ckpt')
    parser.add_argument('--old-prototypes', type=Path, default=Path('outputs/v294_grl_train_prototypes.pt'))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', type=Path, default=Path('outputs/v324_recording_prototypes.pt'))
    parser.add_argument('--report', type=Path, default=Path('reports/v324_train_dispersion.json'))
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError('Refusing to overwrite a completed bank or diagnostic')
    random.seed(823); np.random.seed(823); torch.manual_seed(823)
    train_speakers, dev_speakers = stable_split(Path(args.three_d_root), 180)
    dataset = ShortFullPairDataset(args.three_d_root, train_speakers, args.ocean_root)
    dataset.ocean_root = args.ocean_root
    add_stcmds(dataset, args.stcmds_root)
    add_commonvoice(dataset, args.commonvoice_root)
    add_childmandarin(dataset, args.childmandarin_root)
    rebuild_indexes(dataset)
    old_bank = torch.load(args.old_prototypes, map_location='cpu')
    if (len(dataset.speakers) != 1665 or old_bank['speakers'] != dataset.speakers
            or old_bank['metadata'].get('validation_audio_used') is not False):
        raise RuntimeError('Expected exactly the existing 1665 training identities in order')
    if Path(args.childmandarin_root).resolve().name != 'train':
        raise RuntimeError('ChildMandarin source must be the training split')
    records, recording_paths = [], []
    for index, speaker in enumerate(dataset.speakers):
        paths = sorted(dataset.by_speaker[speaker])
        if len(paths) < 2 or paths[0].resolve() == paths[-1].resolve():
            raise RuntimeError('Two distinct recordings required for ' + speaker)
        selected = [paths[0], paths[-1]]
        if any(not path.is_file() for path in selected):
            raise FileNotFoundError('Missing training recording for ' + speaker)
        recording_paths.append([str(path) for path in selected])
        records.extend((index, recording_index, path)
                       for recording_index, path in enumerate(selected))
    model = DepthTimeGRLW2VBert2(
        args.runtime / 'model/w2vbert2_shards',
        args.runtime / 'model/w2vbert2_config', args.grl_checkpoint,
        args.checkpoint,
    ).cuda().eval().requires_grad_(False)
    print('model=frozen_V302 training_speakers=1665 recordings=3330 batch_size=%d' % args.batch_size,
          flush=True)
    recordings = torch.empty(len(dataset.speakers), 2, 256)
    total_batches = (len(records) + args.batch_size - 1) // args.batch_size
    for batch_index, start in enumerate(range(0, len(records), args.batch_size), 1):
        chunk = records[start:start + args.batch_size]
        # Preserve V294's Python-RNG crop order: sorted identities, first/last,
        # random seed 823, exactly 48,000 samples, no corruption or new views.
        waves = [crop_or_repeat(load_audio(path), 48000) for _, _, path in chunk]
        with torch.autocast('cuda', dtype=torch.float16):
            raw = model(waveforms=torch.stack(waves).cuda()).float()
        if not torch.isfinite(raw).all() or not (raw.norm(dim=1) > 0).all():
            raise RuntimeError('Invalid training recording embedding')
        embedding = F.normalize(raw, dim=1).cpu()
        for row, (speaker_index, recording_index, _) in enumerate(chunk):
            recordings[speaker_index, recording_index] = embedding[row]
        if batch_index % 20 == 0 or batch_index == total_batches:
            print('mining_batches=%d/%d' % (batch_index, total_batches), flush=True)
    means = recordings.mean(dim=1)
    if not torch.isfinite(recordings).all() or not (means.norm(dim=1) > 0).all():
        raise RuntimeError('Invalid or cancelling recording prototypes')
    prototypes = F.normalize(means, dim=1)
    old = F.normalize(old_bank['prototypes'].float(), dim=1)
    if old.shape != prototypes.shape:
        raise RuntimeError('Old classification prototype dimensions differ')
    metadata = {
        'architecture': 'frozen V302 complete depth-time GRL head on S2',
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': file_hash(args.checkpoint),
        'old_prototypes': str(args.old_prototypes),
        'old_prototypes_sha256': file_hash(args.old_prototypes),
        'source': 'existing five-source training split only',
        'seed': 823, 'utterances_per_speaker': 2, 'training_speakers': 1665,
        'samples_per_recording': 48000, 'sample_rate': 16000,
        'selection': 'Lexicographic first and last distinct recording for each existing identity',
        'crop': 'V294 crop_or_repeat, seeded sequential Python RNG, no augmentation',
        'recording_prototypes': 'Each of two 256-dimensional recording vectors is L2-normalized',
        'prototypes': 'L2-normalized mean of the two recording prototypes',
        'validation_audio_used': False, 'validation_trials_used': False,
        'online_feedback_used_for_mining': False, 'model_training_performed': False,
        'three_d_dev_speakers_excluded': len(dev_speakers),
    }
    queries = recordings.flatten(0, 1)
    labels = torch.arange(len(dataset.speakers)).repeat_interleave(2)
    old_margins = class_margins(queries, old, labels)
    new_margins = class_margins(queries, prototypes, labels)
    mean_to_old = class_margins(prototypes, old, torch.arange(len(dataset.speakers)))
    recording_cosine = (recordings[:, 0] * recordings[:, 1]).sum(dim=1)
    domains = sorted({speaker.split(':', 1)[0] for speaker in dataset.speakers})
    report = {'metadata': metadata, 'by_domain': {}, 'limitations': [
        'Training-set geometry diagnostics, not validation accuracy or expected LB.',
        'The new single-center prototype includes each queried recording itself, so its margin improvement is optimistically biased.',
        'Two recordings cannot distinguish session effects, content differences, and stable multimodal identity structure.',
        'No class labels, validation data, candidate scores, or audio selection rules were changed.',
    ]}
    for domain in ['all'] + domains:
        selected = torch.tensor([domain == 'all' or speaker.split(':', 1)[0] == domain
                                 for speaker in dataset.speakers], dtype=torch.bool)
        selected_queries = selected.repeat_interleave(2)
        report['by_domain'][domain] = {
            'speakers': int(selected.sum()),
            'two_recording_cosine': summary(recording_cosine[selected]),
            'v302_recordings_against_old_grl_centers': margins_report(
                tuple(v[selected_queries] for v in old_margins)),
            'v302_recordings_against_new_v302_centers': margins_report(
                tuple(v[selected_queries] for v in new_margins)),
            'margin_change_new_minus_old': summary(
                (new_margins[2] - old_margins[2])[selected_queries]),
            'v302_mean_against_old_grl_centers': margins_report(
                tuple(v[selected] for v in mean_to_old)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'speakers': dataset.speakers, 'prototypes': prototypes,
                'recording_prototypes': recordings, 'recording_paths': recording_paths,
                'metadata': metadata}, args.output)
    report['bank_sha256'] = file_hash(args.output)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'report': str(args.report),
                      'bank_sha256': report['bank_sha256'],
                      'summary': {domain: {
                          'speakers': row['speakers'],
                          'recording_cosine_mean': row['two_recording_cosine']['mean'],
                          'recording_cosine_p05': row['two_recording_cosine']['quantiles']['0.05'],
                          'old_margin_mean': row['v302_recordings_against_old_grl_centers']['own_class_margin']['mean'],
                          'new_margin_mean': row['v302_recordings_against_new_v302_centers']['own_class_margin']['mean'],
                      } for domain, row in report['by_domain'].items()}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
