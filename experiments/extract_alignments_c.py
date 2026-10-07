"""Experiment C step 1: extract MAS phone durations from the trained v6 model.

Uses the FastSpeech2 project's own inference stack ONLY to construct the model
(`Synthesizer` handles config + `_auto_detect_arch` from the state dict), then
runs the training-forward MAS path: `model.forward(tokens, mel_target=...)`
returns `"durations"` whose per-utterance sum equals the valid mel length.

Inputs (strict):
  tokens : dataset/processed/tokens/{uid}.npy   (int32 phoneme IDs)
  mel    : dataset/processed/mel/{uid}.npy      ([80, T], normalized with stats.json)
  lengths: dataset/processed/mel_lengths.json

Output: one JSON per utterance in experiments/alignments_c/ with tokens and
per-phone durations (in mel frames). Resumable: existing outputs are skipped.

Run with cwd = C:/Users/Lenovo/Desktop/fastspeech2.0 (project imports).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("C:/Users/Lenovo/Desktop/fastspeech2.0")
OUT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "alignments_c"
FEATURED = Path(__file__).resolve().parent.parent / "experiments" / "features_wavlm_l6"


def build_model(device: torch.device):
    """Construct the trained v6 model via the project's inference stack."""
    sys.path.insert(0, str(PROJECT))
    from inference.synthesizer import Synthesizer  # noqa: E402

    synth = Synthesizer(
        config_path=str(PROJECT / "configs/experiments/staged_4077_rtx2050_v6.yaml"),
        checkpoint_path=str(PROJECT / "checkpoints_old/staged_4077_rtx2050_v6/latest.pt"),
        tokenizer_path=str(PROJECT / "dataset/processed/phoneme_vocab_approach_a.json"),
        device=device,
        vocoder=None,
        use_g2p=False,
    )
    synth.model.eval()
    return synth


def load_utterance_ids() -> list[str]:
    """All utterances that have WavLM features cached (the Experiment B/C set)."""
    index = json.loads((FEATURED / "index.json").read_text())["utterances"]
    return sorted(index.keys())


def make_batches(sizes: dict[str, int], ids: list[str], batch_size: int) -> list[list[str]]:
    ordered = sorted(ids, key=lambda u: sizes[u])
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]


@torch.no_grad()
def align_batch(synth, uids: list[str], stats: dict, sizes: dict[str, int],
                device: torch.device) -> dict[str, dict]:
    tokens_np = [np.load(PROJECT / "dataset/processed/tokens" / f"{u}.npy").astype(np.int64) for u in uids]
    mels = [np.load(PROJECT / "dataset/processed/mel" / f"{u}.npy").astype(np.float32) for u in uids]
    t_max = max(t.shape[0] for t in tokens_np)
    m_max = max(sizes[u] for u in uids)
    mel_mean = np.array(stats["mel"]["mean"], dtype=np.float32)
    mel_std = np.array(stats["mel"]["std"], dtype=np.float32)

    tokens = torch.zeros(len(uids), t_max, dtype=torch.long)
    src_mask = torch.ones(len(uids), t_max, dtype=torch.bool)  # True = pad
    mel = torch.zeros(len(uids), m_max, 80, dtype=torch.float32)
    mel_mask = torch.ones(len(uids), m_max, dtype=torch.bool)
    for r, (u, t, m) in enumerate(zip(uids, tokens_np, mels)):
        n_t, n_m = t.shape[0], sizes[u]
        tokens[r, :n_t] = torch.from_numpy(t)
        src_mask[r, :n_t] = False
        mel_norm = (m[:, :n_m] - mel_mean[:, None]) / mel_std[:, None]
        mel[r, :n_m] = torch.from_numpy(mel_norm.T)
        mel_mask[r, :n_m] = False

    out = synth.model.forward(
        tokens=tokens.to(device),
        src_mask=src_mask.to(device),
        mel_target=mel.to(device),
        mel_mask=mel_mask.to(device),
        utterance_ids=uids,
        step=0,
    )
    durations = out["durations"].detach().cpu().numpy()
    results = {}
    for r, u in enumerate(uids):
        n_t, n_m = tokens_np[r].shape[0], sizes[u]
        d = durations[r][:n_t].astype(int)
        if int(d.sum()) != n_m:
            raise RuntimeError(
                f"{u}: duration sum {int(d.sum())} != mel length {n_m}"
            )
        results[u] = {
            "tokens": tokens_np[r].tolist(),
            "durations": d.tolist(),
            "n_mel_frames": n_m,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch MAS duration extraction for Experiment C")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="only first N utterances (0 = all)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    stats = json.loads((PROJECT / "dataset/processed/stats.json").read_text())
    sizes = {u: v[1] for u, v in json.loads((PROJECT / "dataset/processed/mel_lengths.json").read_text()).items()}
    ids = load_utterance_ids()
    if args.limit:
        ids = ids[: args.limit]
    missing = [u for u in ids if not (PROJECT / "dataset/processed/tokens" / f"{u}.npy").exists()]
    if missing:
        raise SystemExit(f"missing token files for {len(missing)} utterances, e.g. {missing[:3]}")

    args.out.mkdir(parents=True, exist_ok=True)
    todo = [u for u in ids if not (args.out / f"{u}.json").exists()]
    print(f"{len(ids)} utterances total, {len(todo)} to align", flush=True)

    synth = build_model(device)

    batches = make_batches(sizes, todo, args.batch_size)
    t0 = time.time()
    done = failed = 0
    for bi, batch in enumerate(batches, 1):
        try:
            results = align_batch(synth, batch, stats, sizes, device)
        except Exception as exc:  # retry utterances individually
            print(f"[batch {bi}] failed ({str(exc)[:90]}), retrying individually", flush=True)
            for u in batch:
                try:
                    results = align_batch(synth, [u], stats, sizes, device)
                    for uid, res in results.items():
                        (args.out / f"{uid}.json").write_text(json.dumps(res))
                    done += len(batch)
                except Exception as exc2:
                    print(f"  {u}: FAILED {str(exc2)[:90]}", flush=True)
                    failed += 1
        else:
            for uid, res in results.items():
                (args.out / f"{uid}.json").write_text(json.dumps(res))
            done += len(batch)
        if bi % 25 == 0 or bi == len(batches):
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[{bi}/{len(batches)}] aligned={done} failed={failed} ({rate:.1f} utt/s)", flush=True)

    print(f"done: aligned={done} failed={failed} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
