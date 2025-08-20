#!/bin/bash

# 自动对齐说话人标签与ASR编码器时间维度的批处理脚本
# 作者: AI Assistant
# 日期: 2024

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 默认配置
BASE_DIR="/root/autodl-tmp/joint_sortformer_and_asr_0815"
ASR_MODEL_PATH="${BASE_DIR}/pretrained_models/zh_conformer_transducer_large_bpe_init.nemo"
SCRIPT_PATH="${BASE_DIR}/scripts/auto_align_speaker_labels.py"
SPEAKER_IDS="spk0 spk1 spk2 spk3"
RTTM_FIELD="rttm_filepath"

# 显示帮助信息
show_help() {
    cat << EOF
自动对齐说话人标签工具 - 批处理脚本（支持实际推理）

✨ 新功能：通过实际ASR推理获取编码器时间维度，确保精确对齐

用法: $0 [选项]

选项:
  -h, --help              显示此帮助信息
  -m, --mode MODE         运行模式 (convert|batch_process|batch_process_manifest|update_manifest|generate_dataloader)
  -a, --asr-model PATH    ASR模型路径 (默认: ${ASR_MODEL_PATH})
  -i, --input-manifest PATH    输入manifest文件路径
  -r, --rttm-dir PATH     RTTM文件目录
  -o, --output-npy-dir PATH    输出NPY文件目录
  -u, --output-manifest PATH   输出manifest文件路径
  -s, --speaker-ids "ID0 ID1 ID2 ID3"  说话人ID列表 (默认: "${SPEAKER_IDS}")
  --rttm-path PATH        单个RTTM文件路径 (convert模式)
  --audio-path PATH       单个音频文件路径 (convert模式)
  --output-npy-path PATH  单个输出NPY文件路径 (convert模式)
  --dataloader-output PATH 数据加载器代码输出路径
  --rttm-field FIELD      manifest中RTTM路径字段名 (默认: rttm_filepath)

运行模式说明:
  convert                 - 单文件RTTM到NPY转换（通过实际推理）
  batch_process          - 批量处理整个数据集（传统模式，向后兼容）
  batch_process_manifest - 从manifest批量处理（推荐，支持manifest中的RTTM路径）
  update_manifest        - 更新manifest文件添加NPY路径
  generate_dataloader    - 生成修改后的数据加载器代码

示例:
  # 从manifest批量处理（推荐）
  $0 -m batch_process_manifest -i data/train_with_rttm.json -o data/npy/ -u data/train_with_npy.json
  
  # 传统批量处理（向后兼容）
  $0 -m batch_process -i data/train.json -r data/rttm/ -o data/npy/ -u data/train_with_npy.json
  
  # 单文件转换
  $0 -m convert --rttm-path data/audio1.rttm --audio-path data/audio1.wav --output-npy-path data/audio1.npy
  
  # 更新manifest
  $0 -m update_manifest -i data/train.json -o data/npy/ -u data/train_with_npy.json
  
  # 生成数据加载器
  $0 -m generate_dataloader --dataloader-output modified_dataloader.py

🔥 新功能亮点:
  ✅ 实际推理获取编码器时间维度（不再依赖计算估算）
  ✅ 支持manifest中直接包含RTTM路径
  ✅ 智能RTTM路径推断
  ✅ 详细的处理统计信息
  ✅ 完全向后兼容

EOF
}

# 检查必要的文件和目录
check_prerequisites() {
    print_info "检查前置条件..."
    
    # 检查Python脚本
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        print_error "找不到Python脚本: $SCRIPT_PATH"
        exit 1
    fi
    
    # 检查ASR模型
    if [[ ! -f "$ASR_MODEL_PATH" ]]; then
        print_warning "ASR模型文件不存在: $ASR_MODEL_PATH"
        print_warning "请确保模型路径正确"
    fi
    
    # 检查Python环境
    if ! command -v python &> /dev/null; then
        print_error "找不到Python解释器"
        exit 1
    fi
    
    print_success "前置条件检查完成"
}

# 创建目录
create_directories() {
    if [[ -n "$OUTPUT_NPY_DIR" ]]; then
        print_info "创建输出目录: $OUTPUT_NPY_DIR"
        mkdir -p "$OUTPUT_NPY_DIR"
    fi
    
    if [[ -n "$OUTPUT_MANIFEST" ]]; then
        local output_dir=$(dirname "$OUTPUT_MANIFEST")
        print_info "创建输出目录: $output_dir"
        mkdir -p "$output_dir"
    fi
}

# 运行Python脚本
run_python_script() {
    local cmd=("python" "$SCRIPT_PATH")
    
    # 添加基本参数
    cmd+=("--mode" "$MODE")
    cmd+=("--asr_model_path" "$ASR_MODEL_PATH")
    
    # 根据模式添加特定参数
    case "$MODE" in
        "convert")
            [[ -n "$RTTM_PATH" ]] && cmd+=("--rttm_path" "$RTTM_PATH")
            [[ -n "$AUDIO_PATH" ]] && cmd+=("--audio_path" "$AUDIO_PATH")
            [[ -n "$OUTPUT_NPY_PATH" ]] && cmd+=("--output_npy_path" "$OUTPUT_NPY_PATH")
            ;;
        "batch_process")
            [[ -n "$INPUT_MANIFEST" ]] && cmd+=("--manifest_path" "$INPUT_MANIFEST")
            [[ -n "$RTTM_DIR" ]] && cmd+=("--rttm_dir" "$RTTM_DIR")
            [[ -n "$OUTPUT_NPY_DIR" ]] && cmd+=("--output_npy_dir" "$OUTPUT_NPY_DIR")
            [[ -n "$OUTPUT_MANIFEST" ]] && cmd+=("--output_manifest_path" "$OUTPUT_MANIFEST")
            ;;
        "batch_process_manifest")
            [[ -n "$INPUT_MANIFEST" ]] && cmd+=("--manifest_path" "$INPUT_MANIFEST")
            [[ -n "$OUTPUT_NPY_DIR" ]] && cmd+=("--output_npy_dir" "$OUTPUT_NPY_DIR")
            [[ -n "$OUTPUT_MANIFEST" ]] && cmd+=("--output_manifest_path" "$OUTPUT_MANIFEST")
            [[ -n "$RTTM_FIELD" ]] && cmd+=("--rttm_field" "$RTTM_FIELD")
            ;;
        "update_manifest")
            [[ -n "$INPUT_MANIFEST" ]] && cmd+=("--manifest_path" "$INPUT_MANIFEST")
            [[ -n "$OUTPUT_MANIFEST" ]] && cmd+=("--output_manifest_path" "$OUTPUT_MANIFEST")
            [[ -n "$OUTPUT_NPY_DIR" ]] && cmd+=("--output_npy_dir" "$OUTPUT_NPY_DIR")
            ;;
        "generate_dataloader")
            [[ -n "$DATALOADER_OUTPUT" ]] && cmd+=("--dataloader_output_path" "$DATALOADER_OUTPUT")
            ;;
    esac
    
    # 添加说话人ID
    if [[ -n "$SPEAKER_IDS" ]]; then
        cmd+=("--speaker_ids")
        for spk_id in $SPEAKER_IDS; do
            cmd+=("$spk_id")
        done
    fi
    
    print_info "执行命令: ${cmd[*]}"
    print_info "开始处理..."
    
    # 运行命令
    if "${cmd[@]}"; then
        print_success "处理完成!"
    else
        print_error "处理失败!"
        exit 1
    fi
}

# 显示结果摘要
show_summary() {
    print_info "处理摘要:"
    echo "  模式: $MODE"
    echo "  ASR模型: $ASR_MODEL_PATH"
    
    case "$MODE" in
        "convert")
            echo "  RTTM文件: $RTTM_PATH"
            echo "  音频文件: $AUDIO_PATH"
            echo "  输出NPY: $OUTPUT_NPY_PATH"
            ;;
        "batch_process")
            echo "  输入manifest: $INPUT_MANIFEST"
            echo "  RTTM目录: $RTTM_DIR"
            echo "  输出NPY目录: $OUTPUT_NPY_DIR"
            echo "  输出manifest: $OUTPUT_MANIFEST"
            ;;
        "batch_process_manifest")
            echo "  输入manifest: $INPUT_MANIFEST"
            echo "  输出NPY目录: $OUTPUT_NPY_DIR"
            echo "  输出manifest: $OUTPUT_MANIFEST"
            echo "  RTTM字段名: $RTTM_FIELD"
            ;;
        "update_manifest")
            echo "  输入manifest: $INPUT_MANIFEST"
            echo "  NPY目录: $OUTPUT_NPY_DIR"
            echo "  输出manifest: $OUTPUT_MANIFEST"
            ;;
        "generate_dataloader")
            echo "  数据加载器输出: $DATALOADER_OUTPUT"
            ;;
    esac
    
    echo "  说话人ID: $SPEAKER_IDS"
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -m|--mode)
                MODE="$2"
                shift 2
                ;;
            -a|--asr-model)
                ASR_MODEL_PATH="$2"
                shift 2
                ;;
            -i|--input-manifest)
                INPUT_MANIFEST="$2"
                shift 2
                ;;
            -r|--rttm-dir)
                RTTM_DIR="$2"
                shift 2
                ;;
            -o|--output-npy-dir)
                OUTPUT_NPY_DIR="$2"
                shift 2
                ;;
            -u|--output-manifest)
                OUTPUT_MANIFEST="$2"
                shift 2
                ;;
            -s|--speaker-ids)
                SPEAKER_IDS="$2"
                shift 2
                ;;
            --rttm-path)
                RTTM_PATH="$2"
                shift 2
                ;;
            --audio-path)
                AUDIO_PATH="$2"
                shift 2
                ;;
            --output-npy-path)
                OUTPUT_NPY_PATH="$2"
                shift 2
                ;;
            --dataloader-output)
                DATALOADER_OUTPUT="$2"
                shift 2
                ;;
            --rttm-field)
                RTTM_FIELD="$2"
                shift 2
                ;;
            *)
                print_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 验证参数
validate_arguments() {
    if [[ -z "$MODE" ]]; then
        print_error "必须指定运行模式 (-m|--mode)"
        show_help
        exit 1
    fi
    
    case "$MODE" in
        "convert")
            if [[ -z "$RTTM_PATH" || -z "$AUDIO_PATH" || -z "$OUTPUT_NPY_PATH" ]]; then
                print_error "convert模式需要指定 --rttm-path, --audio-path, --output-npy-path"
                exit 1
            fi
            ;;
        "batch_process")
            if [[ -z "$INPUT_MANIFEST" || -z "$RTTM_DIR" || -z "$OUTPUT_NPY_DIR" || -z "$OUTPUT_MANIFEST" ]]; then
                print_error "batch_process模式需要指定 -i, -r, -o, -u"
                exit 1
            fi
            ;;
        "batch_process_manifest")
            if [[ -z "$INPUT_MANIFEST" || -z "$OUTPUT_NPY_DIR" || -z "$OUTPUT_MANIFEST" ]]; then
                print_error "batch_process_manifest模式需要指定 -i, -o, -u"
                exit 1
            fi
            ;;
        "update_manifest")
            if [[ -z "$INPUT_MANIFEST" || -z "$OUTPUT_MANIFEST" || -z "$OUTPUT_NPY_DIR" ]]; then
                print_error "update_manifest模式需要指定 -i, -u, -o"
                exit 1
            fi
            ;;
        "generate_dataloader")
            if [[ -z "$DATALOADER_OUTPUT" ]]; then
                DATALOADER_OUTPUT="modified_dataloader.py"
                print_info "使用默认数据加载器输出路径: $DATALOADER_OUTPUT"
            fi
            ;;
        *)
            print_error "无效的运行模式: $MODE"
            print_error "支持的模式: convert, batch_process, batch_process_manifest, update_manifest, generate_dataloader"
            exit 1
            ;;
    esac
}

# 主函数
main() {
    print_info "自动对齐说话人标签工具 - 批处理脚本"
    print_info "========================================"
    
    # 解析参数
    parse_arguments "$@"
    
    # 验证参数
    validate_arguments
    
    # 检查前置条件
    check_prerequisites
    
    # 创建必要的目录
    create_directories
    
    # 显示配置摘要
    show_summary
    
    # 确认执行
    echo
    read -p "是否继续执行? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "操作已取消"
        exit 0
    fi
    
    # 运行Python脚本
    run_python_script
    
    print_success "所有操作完成!"
}

# 如果脚本被直接执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi