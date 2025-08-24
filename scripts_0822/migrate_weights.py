#!/usr/bin/env python3
"""
权重迁移脚本：从预训练的EncDecRNNTModel迁移encoder和preprocessor权重到EncDecRNNTBPEModel
"""

import torch
import os
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecRNNTModel, EncDecRNNTBPEModel
from nemo.utils import logging

def migrate_weights(pretrained_model_path, bpe_config_path, tokenizer_dir, output_path):
    """
    迁移权重从预训练模型到BPE模型
    
    Args:
        pretrained_model_path: 预训练模型路径 (.nemo文件)
        bpe_config_path: BPE模型配置文件路径
        tokenizer_dir: tokenizer目录路径
        output_path: 输出模型路径
    """
    
    logging.info(f"Loading pretrained model from {pretrained_model_path}")
    # 加载预训练模型
    pretrained_model = EncDecRNNTModel.restore_from(pretrained_model_path)
    
    logging.info(f"Loading BPE config from {bpe_config_path}")
    # 加载BPE配置
    bpe_config = OmegaConf.load(bpe_config_path)
    
    # 设置tokenizer路径
    bpe_config.model.tokenizer.dir = tokenizer_dir
    
    logging.info("Creating new BPE model")
    # 创建新的BPE模型
    bpe_model = EncDecRNNTBPEModel(cfg=bpe_config.model)
    
    logging.info("Migrating weights...")
    
    # 1. 迁移preprocessor权重
    if hasattr(pretrained_model, 'preprocessor') and hasattr(bpe_model, 'preprocessor'):
        logging.info("Migrating preprocessor weights")
        bpe_model.preprocessor.load_state_dict(pretrained_model.preprocessor.state_dict())
    
    # 2. 迁移encoder权重
    if hasattr(pretrained_model, 'encoder') and hasattr(bpe_model, 'encoder'):
        logging.info("Migrating encoder weights")
        
        # 获取预训练模型和BPE模型的encoder状态字典
        pretrained_encoder_state = pretrained_model.encoder.state_dict()
        bpe_encoder_state = bpe_model.encoder.state_dict()
        
        # 检查兼容性并复制权重
        migrated_keys = []
        skipped_keys = []
        
        for key in pretrained_encoder_state.keys():
            if key in bpe_encoder_state:
                pretrained_shape = pretrained_encoder_state[key].shape
                bpe_shape = bpe_encoder_state[key].shape
                
                if pretrained_shape == bpe_shape:
                    bpe_encoder_state[key] = pretrained_encoder_state[key]
                    migrated_keys.append(key)
                else:
                    logging.warning(f"Shape mismatch for {key}: {pretrained_shape} vs {bpe_shape}")
                    skipped_keys.append(key)
            else:
                logging.warning(f"Key {key} not found in BPE model")
                skipped_keys.append(key)
        
        # 加载迁移后的权重
        bpe_model.encoder.load_state_dict(bpe_encoder_state)
        
        logging.info(f"Successfully migrated {len(migrated_keys)} encoder parameters")
        logging.info(f"Skipped {len(skipped_keys)} parameters due to incompatibility")
        
        if skipped_keys:
            logging.info(f"Skipped keys: {skipped_keys[:10]}...")  # 只显示前10个
    
    # 3. 迁移spec_augment权重（如果存在）
    if hasattr(pretrained_model, 'spec_augmentation') and hasattr(bpe_model, 'spec_augmentation'):
        logging.info("Migrating spec_augment weights")
        try:
            bpe_model.spec_augmentation.load_state_dict(pretrained_model.spec_augmentation.state_dict())
        except Exception as e:
            logging.warning(f"Failed to migrate spec_augment weights: {e}")
    
    logging.info(f"Saving migrated model to {output_path}")
    # 保存新模型
    bpe_model.save_to(output_path)
    
    logging.info("Weight migration completed successfully!")
    
    return bpe_model

def main():
    # 配置路径
    pretrained_model_path = "/root/autodl-tmp/joint_sortformer_and_asr_0815/pretrained_models/conformer_transducer_char_zh.nemo"
    bpe_config_path = "/root/autodl-tmp/joint_sortformer_and_asr_0815/scripts_0822/conformer_transducer_bpe.yaml"
    tokenizer_dir = "/root/autodl-tmp/joint_sortformer_and_asr_0815/tokenizer"  # 需要用户提供tokenizer路径
    output_path = "/root/autodl-tmp/joint_sortformer_and_asr_0815/scripts_0822/conformer_transducer_bpe_migrated.nemo"
    
    # 检查文件是否存在
    if not os.path.exists(pretrained_model_path):
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_model_path}")
    
    if not os.path.exists(bpe_config_path):
        raise FileNotFoundError(f"BPE config not found: {bpe_config_path}")
    
    if not os.path.exists(tokenizer_dir):
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")
    
    # 执行权重迁移
    migrated_model = migrate_weights(
        pretrained_model_path=pretrained_model_path,
        bpe_config_path=bpe_config_path,
        tokenizer_dir=tokenizer_dir,
        output_path=output_path
    )
    
    print(f"\nMigration completed! New model saved to: {output_path}")
    print(f"Model summary:")
    print(f"  - Encoder layers: {migrated_model.encoder.num_layers}")
    print(f"  - Model dimension: {migrated_model.encoder.d_model}")
    print(f"  - Attention heads: {migrated_model.encoder.n_heads}")
    print(f"  - Vocabulary size: {migrated_model.decoder.vocab_size}")

if __name__ == "__main__":
    main()