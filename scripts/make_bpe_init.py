"""
加载 char 版预训练 Conformer-Transducer, 切换 tokenizer 到 BPE（包含说话人 token），并保存为新的 .nemo
"""
import nemo.collections.asr as nemo_asr

# 替换为你的模型名或本地 .nemo
CHAR_MODEL_NAME = "../pretrained_models/stt_zh_conformer_transducer_large.nemo"
BPE_DIR = "../tokenizers/tokenizer_spe_bpe_v1000"  # 训练好的 sentencepiece 目录
OUT_NEMO = "../pretrained_models/zh_conformer_transducer_large_bpe_init.nemo"


def main():
    # 1) 加载 char 预训练模型
    model = nemo_asr.models.EncDecRNNTModel.restore_from(restore_path=CHAR_MODEL_NAME)

    # 2) 切换 tokenizer 到 BPE - 需要先转换为BPE模型
    # 由于原始模型是char模型，我们需要使用不同的方法
    from omegaconf import DictConfig, OmegaConf, open_dict
    import copy
    
    # 获取原始模型配置并深拷贝
    cfg = copy.deepcopy(model.cfg)
    
    # 添加tokenizer配置
    tokenizer_cfg = {
        'dir': BPE_DIR,
        'type': 'bpe'
    }
    
    # 更新配置以支持BPE
    with open_dict(cfg):
        cfg.tokenizer = DictConfig(tokenizer_cfg)
        # 禁用数据加载器以避免初始化时的文件路径问题
        cfg.train_ds = None
        cfg.validation_ds = None
        cfg.test_ds = None
    
    # 创建新的BPE模型（不使用trainer以避免数据加载）
    bpe_model = nemo_asr.models.EncDecRNNTBPEModel(cfg=cfg, trainer=None)
    
    # 复制encoder和preprocessor权重
    bpe_model.encoder.load_state_dict(model.encoder.state_dict())
    bpe_model.preprocessor.load_state_dict(model.preprocessor.state_dict())

    # 3) 保存 .nemo
    bpe_model.save_to(OUT_NEMO)
    print(f"Saved BPE init model to {OUT_NEMO}")

if __name__ == '__main__':
    main()