#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RTTM -> P_gt.npy 对齐脚本（与 ASR 编码器时间轴完全一致）
请参阅画布说明以获取使用方法。
"""
import os
import io
import json
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import soundfile as sf

# NeMo 导入
from nemo.collections.asr.models import ASRModel
torch.set_grad_enabled(False)


def load_nemo_asr(nemo_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    print(f"正在加载 NeMo 模型: {nemo_path}")
    print(f"使用设备: {device}")
    
    # 设置内存优化
    if device == "cpu":
        torch.set_num_threads(1)  # 限制CPU线程数
    
    try:
        model = ASRModel.restore_from(restore_path=nemo_path, map_location=device)
        model = model.to(device)
        model.eval()
        
        # 清理不必要的内存
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.empty_cache()
        
        if not hasattr(model, "preprocessor") or not hasattr(model, "encoder"):
            raise RuntimeError("加载的 NeMo 模型不包含 preprocessor 或 encoder。")
        
        print("模型加载成功")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise


def read_wav(path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return torch.from_numpy(wav), int(sr)


def get_encoder_time_steps(model: ASRModel, wav: torch.Tensor, sr: int, device: str) -> Tuple[torch.Tensor, int, float]:
    wav = wav.to(device)
    wav = wav.unsqueeze(0)
    wav_len = torch.tensor([wav.shape[1]], dtype=torch.int64, device=device)

    proc, proc_len = model.preprocessor(input_signal=wav, length=wav_len)
    enc_out, enc_len = model.encoder(audio_signal=proc, length=proc_len)
    T_enc = int(enc_len[0].item())
    dur_sec = float(wav_len[0].item()) / float(sr)
    return enc_out, T_enc, dur_sec


def parse_rttm_lines(rttm_path: str) -> List[Tuple[str, float, float]]:
    segs = []
    if not os.path.isfile(rttm_path):
        return segs
    with io.open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10 or parts[0].upper() != "SPEAKER":
                continue
            spk = parts[7]
            start = float(parts[3])
            dur = float(parts[4])
            segs.append((spk, start, dur))
    return segs


def build_spk_map(segs: List[Tuple[str, float, float]], k_max: int) -> Dict[str, int]:
    unique = []
    for spk, _, _ in segs:
        if spk not in unique:
            unique.append(spk)
    unique = unique[:k_max]
    return {spk: idx for idx, spk in enumerate(unique)}


def seconds_to_frame_idx(t_sec: float, dur_sec: float, T_enc: int) -> int:
    if dur_sec <= 0 or T_enc <= 0:
        return 0
    x = t_sec / dur_sec * T_enc
    idx = int(math.floor(x + 1e-6))
    return max(0, min(idx, T_enc))


def make_pgt_matrix(T_enc: int, k_max: int, segs: List[Tuple[str, float, float]], spk_map: Dict[str, int], dur_sec: float, soft_edge: bool = False) -> np.ndarray:
    P = np.zeros((k_max, T_enc), dtype=np.float32)
    if T_enc == 0 or dur_sec <= 0:
        return P
    for spk, start, dur in segs:
        if spk not in spk_map:
            continue
        k = spk_map[spk]
        s_idx = seconds_to_frame_idx(start, dur_sec, T_enc)
        e_idx = seconds_to_frame_idx(start + dur, dur_sec, T_enc)
        if e_idx <= s_idx:
            e_idx = min(T_enc, s_idx + 1)
        s_idx = max(0, min(s_idx, T_enc))
        e_idx = max(0, min(e_idx, T_enc))
        if not soft_edge:
            P[k, s_idx:e_idx] = 1.0
        else:
            fade = 3
            P[k, s_idx:e_idx] = 1.0
            for t in range(max(s_idx - fade, 0), s_idx):
                P[k, t] = max(P[k, t], (t - (s_idx - fade)) / max(1, fade))
            for t in range(e_idx, min(e_idx + fade, T_enc)):
                P[k, t] = max(P[k, t], (e_idx + fade - t) / max(1, fade))
    P = np.clip(P, 0.0, 1.0)
    return P


def derive_utt_id(sample: Dict) -> str:
    if "utt_id" in sample and sample["utt_id"]:
        return str(sample["utt_id"])
    audio = sample["audio_filepath"]
    base = os.path.basename(audio)
    return os.path.splitext(base)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo_path", type=str, required=True, help=".nemo 预训练/微调模型路径")
    ap.add_argument("--manifest", type=str, required=True, help="JSON Lines manifest 路径")
    ap.add_argument("--output_dir", type=str, required=True, help="输出 .npy 的目录")
    ap.add_argument("--k_max", type=int, default=4, help="支持的最大说话人数 K_max")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--soft_edge", action="store_true", help="边界做线性淡入淡出")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_nemo_asr(args.nemo_path, device=args.device)

    with io.open(args.manifest, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            audio_fp = sample.get("audio_filepath")
            rttm_fp = sample.get("rttm_filepath")
            if not audio_fp or not os.path.isfile(audio_fp):
                print(f"[WARN] line {ln}: 缺少 audio_filepath 或文件不存在：{audio_fp}")
                continue
            if not rttm_fp or not os.path.isfile(rttm_fp):
                print(f"[WARN] line {ln}: 缺少 rttm_filepath 或文件不存在：{rttm_fp}，将生成全零 P。")

            wav, sr = read_wav(audio_fp)
            try:
                enc_out, T_enc, dur_sec = get_encoder_time_steps(model, wav, sr, device=args.device)
            except Exception as e:
                print(f"[ERROR] line {ln}: 前向失败：{e}")
                continue

            segs = parse_rttm_lines(rttm_fp) if rttm_fp and os.path.isfile(rttm_fp) else []
            spk_map = build_spk_map(segs, args.k_max)

            P = make_pgt_matrix(T_enc=T_enc, k_max=args.k_max, segs=segs, spk_map=spk_map, dur_sec=dur_sec, soft_edge=args.soft_edge)

            utt_id = derive_utt_id(sample)
            out_npy = os.path.join(args.output_dir, f"{utt_id}.npy")
            np.save(out_npy, P)
            print(f"[OK] {utt_id}: T_enc={T_enc}, dur={dur_sec:.2f}s, K_max={args.k_max} -> {out_npy}")

if __name__ == "__main__":
    main()