#!/usr/bin/env python3
"""
测试脚本：验证迁移后的BPE模型是否能正常加载和推理
"""

import torch
import numpy as np
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.utils import logging
import argparse

def test_model_loading(model_path):
    """
    测试模型加载
    
    Args:
        model_path: 模型路径
    """
    
    logging.info(f"Loading model from {model_path}")
    
    try:
        model = EncDecRNNTBPEModel.restore_from(model_path)
        logging.info("✓ Model loaded successfully")
        
        # 打印模型信息
        print(f"\nModel Information:")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Encoder layers: {model.encoder.num_layers}")
        print(f"  - Model dimension: {model.encoder.d_model}")
        print(f"  - Attention heads: {model.encoder.n_heads}")
        print(f"  - Vocabulary size: {model.decoder.vocab_size}")
        print(f"  - Sample rate: {model.sample_rate}")
        
        # 检查tokenizer
        if hasattr(model, 'tokenizer'):
            print(f"  - Tokenizer type: {type(model.tokenizer).__name__}")
            print(f"  - Tokenizer vocab size: {model.tokenizer.vocab_size}")
        
        return model
        
    except Exception as e:
        logging.error(f"✗ Failed to load model: {e}")
        raise

def test_model_inference(model, duration=2.0):
    """
    测试模型推理
    
    Args:
        model: 加载的模型
        duration: 测试音频时长（秒）
    """
    
    logging.info("Testing model inference...")
    
    try:
        # 创建虚拟音频数据
        sample_rate = model.sample_rate
        num_samples = int(duration * sample_rate)
        
        # 生成随机音频信号（模拟真实音频）
        audio_signal = np.random.randn(num_samples).astype(np.float32) * 0.1
        
        # 转换为tensor
        audio_tensor = torch.tensor(audio_signal).unsqueeze(0)  # [1, num_samples]
        audio_length = torch.tensor([num_samples])
        
        logging.info(f"Input audio shape: {audio_tensor.shape}")
        
        # 设置模型为评估模式
        model.eval()
        
        with torch.no_grad():
            # 测试前向传播
            logging.info("Testing forward pass...")
            
            # 预处理
            processed_signal, processed_length = model.preprocessor(
                input_signal=audio_tensor, length=audio_length
            )
            logging.info(f"Preprocessed signal shape: {processed_signal.shape}")
            
            # Encoder
            encoded, encoded_len = model.encoder(
                audio_signal=processed_signal, length=processed_length
            )
            logging.info(f"Encoded signal shape: {encoded.shape}")
            
            # 测试转录
            logging.info("Testing transcription...")
            transcriptions = model.transcribe([audio_signal])
            
            print(f"\nInference Test Results:")
            print(f"  - Input audio duration: {duration}s")
            print(f"  - Input audio samples: {num_samples}")
            print(f"  - Preprocessed shape: {processed_signal.shape}")
            print(f"  - Encoded shape: {encoded.shape}")
            print(f"  - Transcription: {transcriptions[0] if transcriptions else 'None'}")
            
        logging.info("✓ Model inference test passed")
        
    except Exception as e:
        logging.error(f"✗ Model inference test failed: {e}")
        raise

def test_tokenizer(model):
    """
    测试tokenizer功能
    
    Args:
        model: 加载的模型
    """
    
    logging.info("Testing tokenizer...")
    
    try:
        if not hasattr(model, 'tokenizer'):
            logging.warning("Model does not have tokenizer")
            return
        
        tokenizer = model.tokenizer
        
        # 测试文本
        test_texts = [
            "你好世界",
            "这是一个测试",
            "语音识别系统",
            "Hello World",  # 测试英文
            "123456",      # 测试数字
        ]
        
        print(f"\nTokenizer Test Results:")
        print(f"  - Tokenizer vocab size: {tokenizer.vocab_size}")
        
        for text in test_texts:
            try:
                tokens = tokenizer.text_to_tokens(text)
                ids = tokenizer.text_to_ids(text)
                decoded = tokenizer.ids_to_text(ids)
                
                print(f"  - Text: '{text}'")
                print(f"    Tokens: {tokens}")
                print(f"    IDs: {ids}")
                print(f"    Decoded: '{decoded}'")
                print()
                
            except Exception as e:
                print(f"  - Text: '{text}' - Error: {e}")
        
        logging.info("✓ Tokenizer test completed")
        
    except Exception as e:
        logging.error(f"✗ Tokenizer test failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Test migrated BPE model")
    parser.add_argument("--model_path", type=str, 
                       default="/root/autodl-tmp/joint_sortformer_and_asr_0815/scripts_0822/conformer_transducer_bpe_migrated.nemo",
                       help="Path to migrated model")
    parser.add_argument("--audio_duration", type=float, default=2.0,
                       help="Duration of test audio in seconds")
    parser.add_argument("--skip_inference", action="store_true",
                       help="Skip inference test (only test loading)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Testing Migrated BPE Model")
    print("=" * 60)
    
    try:
        # 1. 测试模型加载
        print("\n1. Testing Model Loading...")
        model = test_model_loading(args.model_path)
        
        # 2. 测试tokenizer
        print("\n2. Testing Tokenizer...")
        test_tokenizer(model)
        
        # 3. 测试推理（可选）
        if not args.skip_inference:
            print("\n3. Testing Model Inference...")
            test_model_inference(model, args.audio_duration)
        else:
            print("\n3. Skipping inference test")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! Model is ready for use.")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        raise

if __name__ == "__main__":
    main()