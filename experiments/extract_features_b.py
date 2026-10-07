"""Experiment B step 1: extract frozen WavLM-base layer 6 features.

Extracts hidden_states[6] (content-biased, 50 Hz, H=768) for a fixed subset of
the Nepali FastSpeech2 dataset and caches one fp16 .npy per utterance plus a
JSON index. Resumable: utterances whose cache file already exists are skipped.

Usage:
  python experiments/extract_features_b.py --subset 1200 --test 100
  python experiments/extract_features_b.py --subset 0 --test 100   # test slice only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

DATASET_ROOT = Path("C:/Users/Lenovo/Desktop/fastspeech2.0/dataset")
DEFAULT_OUT = Path("experiments/features_wavlm_l6")
MODEL_ID = "microsoft/wavlm-base"
LAYER = 6
SSL_RATE = 50  # Hz
MAX_SSL_FRAMES = 1000  # ~20 s cap


def load_index(subset: int, test: int) -> dict:
    lengths = json.loads((DATASET_ROOT / "processed/mel_lengths.json").read_text())
    ids = sorted(lengths.keys())
    if subset + test > len(ids):
        raise ValueError(f"requested {subset}+{test} utterances but only {len(ids)} available")
    train_val_ids = ids[:subset]
    test_ids = ids[subset : subset + test]
    split = {
        "train": train_val_ids[: int(len(train_val_ids) * 0.9)],
        "val": train_val_ids[int(len(train_val_ids) * 0.9) :],
        "test": test_ids,
    }
    return split


class WavLMExtractor:
    def __init__(self, device: torch.device):
        from transformers import WavLMModel

        self.device = device
        self.model = WavLMModel.from_pretrained(MODEL_ID).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.extracted = 0

    @torch.no_grad()
    def layer6(self, wav: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
        out = self.model(x, output_hidden_states=True)
        feats = out.hidden_states[LAYER].squeeze(0).T  # [H, T_ssl] -> transposed later
        return feats.T.contiguous().cpu().numpy().astype(np.float16)


def extract(
    ids: list[str],
    extractor: WavLMExtractor,
    out_dir: Path,
    index: dict,
    split_name: str,
    log_every: int = 25,
) -> int:
    import soundfile as sf

    done = 0
    started = time.time()
    for n, utt_id in enumerate(ids, 1):
        cache = out_dir / f"{utt_id}.npy"
        if cache.exists():
            done += 1
            continue
        wav_path = DATASET_ROOT / "raw/wavs" / f"{utt_id}.wav"
        wav, sr = sf.read(wav_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        if wav.size < 400:
            wav = np.pad(wav, (0, 400 - wav.size))
        feats = extractor.layer6(wav)[:MAX_SSL_FRAMES]
        np.save(cache, feats)
        index[utt_id] = {"split": split_name, "frames": int(feats.shape[0]), "dim": int(feats.shape[1])}
        done += 1
        if n % log_every == 0 or n == len(ids):
            rate = done / max(time.time() - started, 1e-6)
            print(
                f"[{split_name}] {done}/{len(ids)} cached, {rate:.2f} utt/s, "
                f"last={utt_id} frames={feats.shape[0]}",
                flush=True,
            )
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=1200)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    split = load_index(args.subset, args.test)
    split_file = args.out / "index.json"
    existing = {}
    if split_file.exists():
        existing = json.loads(split_file.read_text())
        if existing.get("split_config") == split:
            index = existing["utterances"]
            print(f"resuming with {len(index)} cached entries", flush=True)
        else:
            raise SystemExit("index.json exists with a different split config; remove it or change --out")
    else:
        index = {}
        args.out.mkdir(parents=True, exist_ok=True)
        split_file.write_text(
            json.dumps({"split_config": split, "utterances": {}}, indent=1), encoding="utf-8"
        )

    extractor = WavLMExtractor(device)
    total_start = time.time()
    for split_name in ("train", "val", "test"):
        missing = [i for i in split[split_name] if i not in index]
        print(f"{split_name}: {len(missing)} of {len(split[split_name])} to extract", flush=True)
        extract(missing, extractor, args.out, index, split_name, args.log_every)
        split_file.write_text(
            json.dumps({"split_config": split, "utterances": index}, indent=1), encoding="utf-8"
        )

    frames = [v["frames"] for v in index.values()]
    print(
        f"done in {time.time() - total_start:.0f}s: {len(index)} utterances, "
        f"frames min/mean/max = {min(frames)}/{sum(frames)/len(frames):.0f}/{max(frames)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
