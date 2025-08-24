#!/usr/bin/env python3
"""
Tokenizer设置脚本：为BPE模型创建和配置tokenizer
"""

import os
import argparse
from pathlib import Path
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.utils import logging

def create_sample_tokenizer(output_dir, vocab_size=1000):
    """
    创建一个示例BPE tokenizer
    
    Args:
        output_dir: 输出目录
        vocab_size: 词汇表大小
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 示例中文文本用于训练tokenizer
    sample_text = """
    你好世界
    这是一个测试文本
    用于训练中文BPE分词器
    包含常用的中文词汇和短语
    希望能够正确处理中文字符
    自然语言处理技术
    语音识别系统
    深度学习模型
    人工智能应用
    机器学习算法
    """
    
    # 保存示例文本
    text_file = os.path.join(output_dir, "sample_text.txt")
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    logging.info(f"Creating BPE tokenizer with vocab_size={vocab_size}")
    
    # 创建tokenizer
    tokenizer = SentencePieceTokenizer(
        model_path=None,  # 将会自动创建
        vocab_size=vocab_size,
        sample_size=-1,
        do_lower_case=False,
        user_defined_symbols=['<unk>', '<pad>', '<s>', '</s>'],
    )
    
    # 训练tokenizer
    model_path = os.path.join(output_dir, "tokenizer.model")
    tokenizer.train(
        text=[text_file],
        vocab_size=vocab_size,
        model_path=model_path,
        model_type='bpe',
        character_coverage=0.9999,
        input_sentence_size=1000000,
        shuffle_input_sentence=True
    )
    
    logging.info(f"Tokenizer saved to {model_path}")
    
    # 测试tokenizer
    test_text = "你好世界，这是一个测试。"
    tokens = tokenizer.text_to_tokens(test_text)
    ids = tokenizer.text_to_ids(test_text)
    
    print(f"\nTokenizer test:")
    print(f"Original text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {ids}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    return model_path

def setup_existing_tokenizer(tokenizer_path, output_dir):
    """
    设置已存在的tokenizer
    
    Args:
        tokenizer_path: 现有tokenizer模型路径
        output_dir: 输出目录
    """
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制tokenizer文件
    import shutil
    output_path = os.path.join(output_dir, "tokenizer.model")
    shutil.copy2(tokenizer_path, output_path)
    
    logging.info(f"Tokenizer copied to {output_path}")
    
    # 测试tokenizer
    tokenizer = SentencePieceTokenizer(model_path=output_path)
    
    test_text = "你好世界，这是一个测试。"
    tokens = tokenizer.text_to_tokens(test_text)
    ids = tokenizer.text_to_ids(test_text)
    
    print(f"\nTokenizer test:")
    print(f"Original text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {ids}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Setup BPE tokenizer for ASR model")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/joint_sortformer_and_asr_0815/tokenizer",
                       help="Output directory for tokenizer")
    parser.add_argument("--existing_tokenizer", type=str, default=None,
                       help="Path to existing tokenizer model")
    parser.add_argument("--vocab_size", type=int, default=1000,
                       help="Vocabulary size for new tokenizer")
    
    args = parser.parse_args()
    
    if args.existing_tokenizer:
        # 使用现有tokenizer
        logging.info(f"Setting up existing tokenizer from {args.existing_tokenizer}")
        tokenizer_path = setup_existing_tokenizer(args.existing_tokenizer, args.output_dir)
    else:
        # 创建新tokenizer
        logging.info(f"Creating new tokenizer with vocab_size={args.vocab_size}")
        tokenizer_path = create_sample_tokenizer(args.output_dir, args.vocab_size)
    
    print(f"\nTokenizer setup completed!")
    print(f"Tokenizer directory: {args.output_dir}")
    print(f"Tokenizer model: {tokenizer_path}")
    print(f"\nYou can now use this tokenizer directory in your BPE model configuration.")
    print(f"Update the config file with: tokenizer.dir = '{args.output_dir}'")

if __name__ == "__main__":
    main()