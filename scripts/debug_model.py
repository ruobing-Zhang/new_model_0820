#!/usr/bin/env python
# -*- coding: utf-8 -*-

import traceback
from model_with_spk_inject import create_model_from_bpe_checkpoint

try:
    model = create_model_from_bpe_checkpoint(
        '../pretrained_models/zh_conformer_transducer_large_bpe_init.nemo', 
        pgt_dir='./output_npy'
    )
    print("模型创建成功!")
except Exception as e:
    print(f"错误: {e}")
    traceback.print_exc()