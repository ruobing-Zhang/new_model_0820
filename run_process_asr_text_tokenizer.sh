python NeMo-main/scripts/tokenizers/process_asr_text_tokenizer.py \
    --manifest="data/M8013_multispeaker_manifest_train_joint_no_punc.json" \
    --data_root="./tokenizers" \
    --vocab_size=1000 \
    --tokenizer="spe" \
    --spe_user_defined_symbols "<|spk0|>" "<|spk1|>" "<|spk2|>" "<|spk3|>" \
    --log