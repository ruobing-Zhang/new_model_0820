# collate_with_spk.py
import torch
import numpy as np
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence

def custom_collate_with_spk(batch):
    """
    batch: 来自 ManifestSpeakerDataset.__getitem__ 返回的字典列表
      - audio_filepath (str)
      - text (str)
      - pgt_npy (str)
    输出：
      - input_signal: [B, T_max]
      - input_signal_length: [B]
      - text: List[str]
      - spk_labels: [B, K, T_max_spk]（按时间维做 0-pad；注入时再插值到 T_enc）
      - ids: List[str]
    """
    wavs = []
    wav_lens = []
    texts = []
    ids = []
    spks = []

    for ex in batch:
        wav, sr = sf.read(ex["audio_filepath"], dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(-1)
        wavs.append(torch.tensor(wav, dtype=torch.float32))
        wav_lens.append(len(wav))
        texts.append(ex["text"])
        ids.append(ex.get("utt_id", ex["audio_filepath"].split("/")[-1].split(".")[0]))

        P = np.load(ex["pgt_npy"]).astype("float32")  # [K, T_any] (与某个 encoder 对齐或待对齐)
        spks.append(torch.tensor(P, dtype=torch.float32))

    # pad audio
    input_signal = pad_sequence(wavs, batch_first=True)  # [B, T_max]
    input_signal_length = torch.tensor(wav_lens, dtype=torch.int64)

    # pad speaker matrices on time dim
    K = max(p.shape[0] for p in spks)
    T_spk_max = max(p.shape[1] for p in spks)
    spk_labels = torch.zeros(len(spks), K, T_spk_max, dtype=torch.float32)
    for i, p in enumerate(spks):
        spk_labels[i, :p.shape[0], :p.shape[1]] = p

    return {
        "input_signal": input_signal,
        "input_signal_length": input_signal_length,
        "text": texts,
        "spk_labels": spk_labels,
        "ids": ids,
    }
