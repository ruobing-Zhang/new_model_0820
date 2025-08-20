# train_multispeaker_asr_adapter.py
import os
import argparse
from typing import List

import torch
import pytorch_lightning as pl

from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis  # 可选：推理时用
from nemo.collections.asr.models import ASRModel  # 仅用于加载 .nemo 检查

from model_with_spk_inject import RNNTWithSpkInject, ManifestSpeakerDataset
from collate_with_spk import custom_collate_with_spk


# ---------- Lightning 封装（计算 RNNT loss） ----------
class LitRNNTWithSpk(pl.LightningModule):
    def __init__(self, nemo_model_path: str, train_manifest: str, val_manifest: str,
                 K_max: int = 4, alpha_init: float = 0.1,
                 lr: float = 1e-4, weight_decay: float = 1e-3,
                 batch_size: int = 8, num_workers: int = 4,
                 use_adapter: bool = False, adapter_cfg: dict = None):
        super().__init__()
        self.save_hyperparameters()

        # 1) 从 .nemo 恢复基础 RNNT 模型配置与权重
        base = ASRModel.restore_from(nemo_model_path, map_location='cpu')
        base.eval()

        # 2) 创建配置副本并清理数据加载器配置
        import copy
        from omegaconf import open_dict
        cfg = copy.deepcopy(base.cfg)
        with open_dict(cfg):
            if 'train_ds' in cfg:
                cfg.train_ds = None
            if 'validation_ds' in cfg:
                cfg.validation_ds = None
            if 'test_ds' in cfg:
                cfg.test_ds = None

        # 3) 用相同 cfg 初始化我们带注入的模型，并传递base_model
        self.model = RNNTWithSpkInject(cfg=cfg, trainer=self, K_max=K_max, alpha_init=alpha_init, base_model=base)
        
        # 4) 复制其他权重
        missing, unexpected = self.model.load_state_dict(base.state_dict(), strict=False)
        if missing:
            logging.info(f"[load_state_dict] missing keys: {missing}")
        if unexpected:
            logging.info(f"[load_state_dict] unexpected keys: {unexpected}")

        # 3) （可选）在 encoder 上加 adapter（只需两步）
        if use_adapter:
            adapter_supported = hasattr(self.model.encoder, "add_adapter")
            logging.info(f"Encoder adapter 支持状态: {adapter_supported}")
            
            if not adapter_supported:
                logging.warning("当前 NeMo 版本 encoder 不支持 add_adapter(); 使用 fallback 注入方法。")
                # 使用fallback方法：只解冻说话人注入相关的参数
                for name, param in self.model.named_parameters():
                    if any(key in name for key in ['alpha', 'Gamma', 'spk']):
                        param.requires_grad = True
                        logging.info(f"解冻参数: {name}")
                    else:
                        param.requires_grad = False
            else:
                name = adapter_cfg.get("name", "enc_adapter")
                cfg = adapter_cfg.get("cfg", {"in_features": self.model.M, "dim": self.model.M // 2})
                logging.info(f"添加 adapter: {name}, 配置: {cfg}")
                self.model.encoder.add_adapter(name=name, cfg=cfg)
                if hasattr(self.model.encoder, "set_enabled_adapters"):
                    self.model.encoder.set_enabled_adapters(enabled=False)
                self.model.encoder.set_enabled_adapters(name, enabled=True)
                    
            # 冻结所有参数
            self.model.freeze()
            logging.info("已冻结所有模型参数")
            
            # 解冻 encoder adapter 参数
            if adapter_supported:
                if hasattr(self.model.encoder, "unfreeze_enabled_adapters"):
                    self.model.encoder.unfreeze_enabled_adapters()
                    logging.info("使用 unfreeze_enabled_adapters 解冻 adapter")
                else:
                    for n, p in self.model.named_parameters():
                        if "adapter" in n:
                            p.requires_grad = True
                            logging.info(f"手动解冻 adapter 参数: {n}")
            
            # 解冻说话人注入相关参数
            for name, param in self.model.named_parameters():
                if any(key in name for key in ['alpha', 'Gamma', 'spk']):
                    param.requires_grad = True
                    logging.info(f"解冻说话人注入参数: {name}")
            
            # 解冻解码器所有参数
            for name, param in self.model.decoder.named_parameters():
                param.requires_grad = True
                logging.info(f"解冻解码器参数: {name}")
            
            # 解冻joint网络所有参数
            for name, param in self.model.joint.named_parameters():
                param.requires_grad = True
                logging.info(f"解冻joint网络参数: {name}")
            
            # 统计可训练参数数量
            total_params = 0
            trainable_params = 0
            for name, param in self.model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            
            logging.info(f"=== 参数统计 ===")
            logging.info(f"总参数数量: {total_params:,}")
            logging.info(f"可训练参数数量: {trainable_params:,}")
            logging.info(f"可训练参数比例: {trainable_params/total_params*100:.2f}%")
            logging.info(f"===============")

        # 4) Data
        self.train_ds = ManifestSpeakerDataset(train_manifest)
        self.val_ds = ManifestSpeakerDataset(val_manifest)
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 5) 优化器超参
        self.lr = lr
        self.weight_decay = weight_decay

    # ------- dataloaders -------
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, collate_fn=custom_collate_with_spk
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, collate_fn=custom_collate_with_spk
        )

    # ------- 优化器 -------
    def configure_optimizers(self):
        # 按模块分类统计参数
        encoder_params = []
        decoder_params = []
        joint_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name.startswith('encoder.'):
                    encoder_params.append((name, param))
                elif name.startswith('decoder.'):
                    decoder_params.append((name, param))
                elif name.startswith('joint.'):
                    joint_params.append((name, param))
                else:
                    other_params.append((name, param))
        
        # 详细统计信息
        logging.info(f"=== 优化器参数详情 ===")
        logging.info(f"Encoder 可训练参数: {len(encoder_params)} 个")
        for name, param in encoder_params:
            logging.info(f"  - {name}: {param.shape} ({param.numel():,} 参数)")
        
        logging.info(f"Decoder 可训练参数: {len(decoder_params)} 个")
        for name, param in decoder_params:
            logging.info(f"  - {name}: {param.shape} ({param.numel():,} 参数)")
        
        logging.info(f"Joint 可训练参数: {len(joint_params)} 个")
        for name, param in joint_params:
            logging.info(f"  - {name}: {param.shape} ({param.numel():,} 参数)")
        
        logging.info(f"其他可训练参数: {len(other_params)} 个")
        for name, param in other_params:
            logging.info(f"  - {name}: {param.shape} ({param.numel():,} 参数)")
        
        all_params = encoder_params + decoder_params + joint_params + other_params
        
        if not all_params:
            logging.error("没有找到需要优化的参数！")
            # 作为fallback，至少优化alpha参数
            if hasattr(self.model, 'alpha'):
                self.model.alpha.requires_grad = True
                all_params.append(('alpha', self.model.alpha))
                logging.info(f"Fallback: 添加alpha参数到优化器")
        
        if not all_params:
            raise ValueError("仍然没有找到可优化的参数")
        
        total_trainable = sum(p.numel() for _, p in all_params)
        logging.info(f"优化器总计可训练参数: {total_trainable:,}")
        logging.info(f"=====================")
            
        optim = torch.optim.AdamW((p for _, p in all_params), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=1000)
        return {"optimizer": optim, "lr_scheduler": sched}

    # ------- RNNT 损失的最小实现 -------
    def _texts_to_targets(self, texts: List[str]):
        """
        使用 NeMo 的 tokenizer 将文本转为标签 ids，并打包成稠密张量。
        返回：targets [B, U_max], target_length [B]
        """
        tokenizer = self.model.tokenizer
        logging.info(f"[DEBUG] tokenizer类型: {type(tokenizer)}")
        
        # 处理不同类型的tokenizer
        if hasattr(tokenizer, 'text_to_ids'):
            ids_list = [tokenizer.text_to_ids(t) for t in texts]
        elif hasattr(tokenizer, 'labels'):
            # 如果是vocabulary配置，使用简单的字符级tokenization
            labels = list(tokenizer.labels) if hasattr(tokenizer, 'labels') else list(tokenizer)
            char_to_id = {char: i for i, char in enumerate(labels)}
            ids_list = [[char_to_id.get(char, 0) for char in text] for text in texts]
            logging.info(f"[DEBUG] 使用字符级tokenization，词汇表大小: {len(labels)}")
        else:
            # 最简单的fallback：使用字符的ASCII值
            ids_list = [[ord(char) % 1000 for char in text] for text in texts]
            logging.warning(f"[DEBUG] 使用ASCII fallback tokenization")
            
        target_length = torch.tensor([len(x) for x in ids_list], dtype=torch.int32, device=self.device)
        U_max = max(l.item() for l in target_length) if target_length.numel() > 0 else 1
        targets = torch.zeros(len(ids_list), U_max, dtype=torch.int64, device=self.device)
        for i, ids in enumerate(ids_list):
            if len(ids) > 0:
                targets[i, :len(ids)] = torch.tensor(ids, dtype=torch.int64, device=self.device)
        return targets, target_length

    def _rnnt_forward_and_loss(self, batch) -> torch.Tensor:
        """
        计算 RNNT 损失：
          1) 预处理+编码+注入 → enc_out, enc_len
          2) 文本转 ids → targets, target_len
          3) 解码器前向 → dec_out
          4) joint → logits
          5) RNNT loss
        """
        input_signal = batch["input_signal"].to(self.device)
        input_signal_length = batch["input_signal_length"].to(self.device)
        spk_labels = batch["spk_labels"].to(self.device)
        texts = batch["text"]

        # (1) encode + injection
        enc_out, enc_len = self.model.encode_with_injection(input_signal, input_signal_length, spk_labels)
        # enc_out: [B,T,M], enc_len: [B]

        # (2) texts -> targets
        targets, target_length = self._texts_to_targets(texts)  # [B,U], [B]
        print(f"[DEBUG] targets.shape: {targets.shape}, target_length: {target_length}")

        # (3) prediction network
        # NeMo 的 RNNT decoder 支持 (targets, target_length) 输入
        decoder_output = self.model.decoder(targets=targets, target_length=target_length)
        print(f"[DEBUG] targets.shape = {targets.shape}")
        print(f"[DEBUG] target_length = {target_length}")
        print(f"[DEBUG] target_length.max() = {target_length.max()}")
        print(f"[DEBUG] decoder_output type: {type(decoder_output)}, length: {len(decoder_output) if isinstance(decoder_output, (tuple, list)) else 'N/A'}")
        
        if isinstance(decoder_output, tuple):
            if len(decoder_output) == 2:
                dec_out, dec_len = decoder_output
                print(f"[DEBUG] 解码器输出 (2元组): dec_out.shape={dec_out.shape}, dec_len={dec_len}")
            elif len(decoder_output) == 3:
                dec_out, dec_len, _ = decoder_output  # 忽略第三个返回值
                print(f"[DEBUG] 解码器输出 (3元组): dec_out.shape={dec_out.shape}, dec_len={dec_len}")
            else:
                print(f"[DEBUG] Unexpected decoder output length: {len(decoder_output)}")
                dec_out = decoder_output[0]
                dec_len = target_length
                print(f"[DEBUG] 使用第一个元素: dec_out.shape={dec_out.shape}")
        else:
            dec_out = decoder_output
            dec_len = target_length  # 使用输入的长度作为fallback
            print(f"[DEBUG] 单一输出: dec_out.shape={dec_out.shape}")
        
        # 修复解码器输出形状：从 [B, D, U] 转换为 [B, U, D]
        if dec_out.dim() == 3 and dec_out.shape[1] > dec_out.shape[2]:
            print(f"[DEBUG] 转换解码器输出形状: {dec_out.shape} -> ", end="")
            dec_out = dec_out.transpose(1, 2).contiguous()  # [B, D, U] -> [B, U, D]
            print(f"{dec_out.shape}")
        
        print(f"[DEBUG] Final dec_out shape: {dec_out.shape} (expected: [B, U, D])")
        # dec_out: [B,U,D]  dec_len: [B]

        # (4) joint network
        # 简化joint网络调用，只传递核心参数
        print(f"[DEBUG] 调用joint前: enc_out.shape={enc_out.shape}, dec_out.shape={dec_out.shape}")
        print(f"[DEBUG] enc_len={enc_len}, target_length={target_length}")
        
        print(f"[DEBUG] 使用原始enc_len: {enc_len}")
        
        # 使用正确的 joint 网络调用方式
        # enc_out: [B, T, M] -> 需要转换为 [B, T, H1] 格式
        # dec_out: [B, U, H2] 格式正确
        print(f"[DEBUG] 调用 joint.joint() 方法")
        print(f"[DEBUG] enc_out.shape: {enc_out.shape}, dec_out.shape: {dec_out.shape}")
        
        # 直接使用 joint() 方法
        logits = self.model.joint.joint(enc_out, dec_out)
        print(f"[DEBUG] joint 输出 logits.shape: {logits.shape}")
        # logits: [B,T,U,V]
        
        # 将 logits 转换为 log_probs
        import torch.nn.functional as F
        log_probs = F.log_softmax(logits, dim=-1)

        # (5) RNNT loss
        loss = self.model.loss(log_probs=log_probs, targets=targets, target_lengths=target_length, input_lengths=enc_len)
        return loss

    # ------- Lightning hooks -------
    def training_step(self, batch, batch_idx):
        loss = self._rnnt_forward_and_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch["input_signal"].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._rnnt_forward_and_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch["input_signal"].size(0))
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo_model", type=str, required=True, help=".nemo 模型路径（如 zh_conformer_transducer_large_bpe_init.nemo）")
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--K_max", type=int, default=4)
    parser.add_argument("--alpha_init", type=float, default=0.1)
    parser.add_argument("--use_adapter", action="store_true")
    args = parser.parse_args()

    pl.seed_everything(1234)

    model = LitRNNTWithSpk(
        nemo_model_path=args.nemo_model,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        K_max=args.K_max,
        alpha_init=args.alpha_init,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_adapter=args.use_adapter,
        adapter_cfg={"name": "enc_adapter", "cfg": {"in_features": 512, "dim": 256}},
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        log_every_n_steps=10,
        val_check_interval=0.25,
        gradient_clip_val=1.0,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
