"""Rebuild training labels from published ST-CMDS identity namespaces."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mine_v324_recording_prototypes import (
    file_hash, summary, stable_split, ShortFullPairDataset,
    add_stcmds, add_commonvoice, add_childmandarin, rebuild_indexes,
    DepthTimeGRLW2VBert2, load_audio, crop_or_repeat,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path,
                        default=Path('checkpoints/pretrained/r45_runtime'))
    parser.add_argument('--three-d-root', type=Path,
                        default=Path('data/processed/3dspeaker'))
    parser.add_argument('--ocean-root', type=Path,
                        default=Path('data/processed/speechocean'))
    parser.add_argument('--stcmds-root', type=Path,
                        default=Path('data/processed/stcmds/ST-CMDS-20170001_1-OS'))
    parser.add_argument('--commonvoice-root', type=Path,
                        default=Path('data/processed/commonvoice17-train'))
    parser.add_argument('--childmandarin-root', type=Path,
                        default=Path('data/raw/childmandarin/train'))
    parser.add_argument('--grl-checkpoint', type=Path, default=None)
    parser.add_argument('--teacher-checkpoint', type=Path,
                        default=Path('checkpoints/training/v302_final.ckpt'))
    parser.add_argument('--legacy-prototypes', type=Path,
                        default=Path('checkpoints/training/v324_recording_prototypes.pt'))
    parser.add_argument('--output', type=Path,
                        default=Path('outputs/v328_corrected_prototypes.pt'))
    parser.add_argument('--report', type=Path,
                        default=Path('reports/v328_corrected_prototypes.json'))
    args = parser.parse_args()
    output = args.output
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    grl_checkpoint = args.grl_checkpoint or (
        args.runtime / 'model/w2vbert2_grl_head.ckpt'
    )
    random.seed(823); np.random.seed(823); torch.manual_seed(823)
    train, dev = stable_split(args.three_d_root, 180)
    data = ShortFullPairDataset(args.three_d_root, train, args.ocean_root)
    data.ocean_root = str(args.ocean_root)
    count, items = add_stcmds(data, args.stcmds_root)
    add_commonvoice(data, args.commonvoice_root)
    add_childmandarin(data, args.childmandarin_root); rebuild_indexes(data)
    assert count == 855 and items == 102600 and len(data.speakers) == 2077
    legacy = torch.load(args.legacy_prototypes, map_location='cpu')
    assert legacy['metadata']['validation_audio_used'] is False
    previous = {s: i for i, s in enumerate(legacy['speakers'])}
    assert {s for s in data.speakers if not s.startswith('stcmds:')} == {s for s in previous if not s.startswith('stcmds:')}
    vectors = torch.empty(len(data.speakers), 2, 256)
    paths, tasks = [], []
    for i, speaker in enumerate(data.speakers):
        if not speaker.startswith('stcmds:'):
            vectors[i] = legacy['recording_prototypes'][previous[speaker]]
            paths.append(legacy['recording_paths'][previous[speaker]])
        else:
            selected = sorted(data.by_speaker[speaker])
            assert len(selected) == 120
            selected = [selected[0], selected[-1]]
            assert selected[0] != selected[1]
            paths.append([str(p) for p in selected])
            tasks.extend((i, j, p) for j, p in enumerate(selected))
    model = DepthTimeGRLW2VBert2(
        args.runtime / 'model/w2vbert2_shards',
        args.runtime / 'model/w2vbert2_config',
        grl_checkpoint,
        args.teacher_checkpoint,
    ).cuda().eval().requires_grad_(False)
    for start in range(0, len(tasks), 16):
        chunk = tasks[start:start+16]
        waves = torch.stack([crop_or_repeat(load_audio(p), 48000) for _, _, p in chunk]).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            raw = model(waveforms=waves).float()
        assert torch.isfinite(raw).all() and (raw.norm(dim=1)>0).all()
        values = F.normalize(raw, dim=1).cpu()
        for row, (i, j, _) in enumerate(chunk): vectors[i, j] = values[row]
        if start % 320 == 0: print('corrected_recordings=%d/%d' % (start+len(chunk),len(tasks)),flush=True)
    metadata = {'training_speakers':len(data.speakers),'stcmds_speakers':count,
        'identity_schema':'ST-CMDS Pxxxxx plus Android/IOS namespace (A/I)',
        'identity_evidence':'reports/stcmds_identity_audit.json and https://www.openslr.org/38/',
        'validation_audio_used':False,'validation_trials_used':False,'online_feedback_used_for_mining':False,
        'source':'V302 representations of corrected training identities; unchanged non-ST recording vectors reused exactly',
        'legacy_bank_sha256':file_hash(args.legacy_prototypes),
        'teacher_checkpoint_sha256':file_hash(args.teacher_checkpoint),
        'seed':823,'samples_per_recording':48000,'utterances_per_speaker':2,
        'newly_extracted_recordings':len(tasks),'three_d_dev_speakers_excluded':len(dev)}
    torch.save({'speakers':data.speakers,'prototypes':F.normalize(vectors.mean(1),dim=1),
        'recording_prototypes':vectors,'recording_paths':paths,'metadata':metadata}, output)
    cosine=(vectors[:,0]*vectors[:,1]).sum(-1)
    report={'metadata':metadata,'bank_sha256':file_hash(output),'within_identity_cosine':{}}
    for domain in ('3d','child','cv','ocean','stcmds'):
        mask=torch.tensor([s.startswith(domain+':') for s in data.speakers])
        report['within_identity_cosine'][domain]=summary(cosine[mask])
    args.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__': main()
