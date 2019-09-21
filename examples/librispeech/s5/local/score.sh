#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

model=/aifs/users/tx078/tools/neural_sp_parallel/examples/librispeech/s5/exp/librispeech/asr/train_960_wpbpe30000/blstm512H5L_drop8_lstm1024H1L_location_adam_lr0.002_bs30_ss0.2_ls0.1_4GPU/model.epoch-17
model1=
model2=
model3=
model_bwd=
gpu=
stdout=false

### path to save preproecssed data
data=/aifs/users/tx078/data/asr/datas/librispeech

unit=
metric=edit_distance
batch_size=1
beam_width=5
min_len_ratio=0.0
max_len_ratio=1.0
length_penalty=0.0
coverage_penalty=0.0
coverage_threshold=0.0
gnmt_decoding=false
eos_threshold=1.5
lm=
lm_bwd=
lm_weight=0.3
lm_usage=shallow_fusion
ctc_weight=0.0  # 1.0 for joint CTC-attention means decoding with CTC
resolving_unk=false
fwd_bwd_attention=false
bwd_attention=false
reverse_lm_rescoring=false
asr_state_carry_over=false
lm_state_carry_over=true
n_caches=0
cache_theta_speech=1.5
cache_lambda_speech=0.1
cache_theta_lm=0.1
cache_lambda_lm=0.1
cache_type=lm_fifo
oracle=false

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ -z ${gpu} ]; then
    echo "Error: set GPU number." 1>&2
    echo "Usage: local/score.sh --gpu 0" 1>&2
    exit 1
fi
gpu=$(echo ${gpu} | cut -d "," -f 1)

for set in dev_clean dev_other test_clean test_other; do
    recog_dir=$(dirname ${model})/decode_${set}_beam${beam_width}_lp${length_penalty}_cp${coverage_penalty}_${min_len_ratio}_${max_len_ratio}
    if [ ! -z ${unit} ]; then
        recog_dir=${recog_dir}_${unit}
    fi
    if [ ${metric} != 'edit_distance' ]; then
        recog_dir=${recog_dir}_${metric}
    fi
    if [ ! -z ${lm} ] && [ ${lm_weight} != 0 ]; then
        recog_dir=${recog_dir}_lm${lm_weight}_${lm_usage}
    fi
    if [ ${ctc_weight} != 0.0 ]; then
        recog_dir=${recog_dir}_ctc${ctc_weight}
    fi
    if [ ${gnmt_decoding} = true ]; then
        recog_dir=${recog_dir}_gnmt
    fi
    if [ ${resolving_unk} = true ]; then
        recog_dir=${recog_dir}_resolvingOOV
    fi
    if [ ${fwd_bwd_attention} = true ]; then
        recog_dir=${recog_dir}_fwdbwd
    fi
    if [ ${bwd_attention} = true ]; then
        recog_dir=${recog_dir}_bwd
    fi
    if [ ${reverse_lm_rescoring} = true ]; then
        recog_dir=${recog_dir}_revLM
    fi
    if [ ${asr_state_carry_over} = true ]; then
        recog_dir=${recog_dir}_ASRcarryover
    fi
    if [ ! -z ${lm} ] && [ ${lm_weight} != 0 ] && [ ${lm_state_carry_over} = true ]; then
        recog_dir=${recog_dir}_LMcarryover
    fi
    if [ ${n_caches} != 0 ]; then
        recog_dir=${recog_dir}_${cache_type}cache${n_caches}
    fi
    if [ ${oracle} = true ]; then
        recog_dir=${recog_dir}_oracle
    fi
    if [ ! -z ${model3} ]; then
        recog_dir=${recog_dir}_ensemble4
    elif [ ! -z ${model2} ]; then
        recog_dir=${recog_dir}_ensemble3
    elif [ ! -z ${model1} ]; then
        recog_dir=${recog_dir}_ensemble2
    fi
    mkdir -p ${recog_dir}

    if [ $(echo ${model} | grep '960') ]; then
        recog_set=${data}/dataset/${set}_960_wpbpe30000.tsv
    elif [ $(echo ${model} | grep '460') ]; then
        recog_set=${data}/dataset/${set}_460_wpbpe10000.tsv
    elif [ $(echo ${model} | grep '100') ]; then
        recog_set=${data}/dataset/${set}_100_wpbpe1000.tsv
    fi

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/eval.py \
        --recog_sets ${recog_set} \
        --recog_dir ${recog_dir} \
        --recog_unit ${unit} \
        --recog_metric ${metric} \
        --recog_model ${model} ${model1} ${model2} ${model3} \
        --recog_model_bwd ${model_bwd} \
        --recog_batch_size ${batch_size} \
        --recog_beam_width ${beam_width} \
        --recog_max_len_ratio ${max_len_ratio} \
        --recog_min_len_ratio ${min_len_ratio} \
        --recog_length_penalty ${length_penalty} \
        --recog_coverage_penalty ${coverage_penalty} \
        --recog_coverage_threshold ${coverage_threshold} \
        --recog_gnmt_decoding ${gnmt_decoding} \
        --recog_eos_threshold ${eos_threshold} \
        --recog_lm ${lm} \
        --recog_lm_bwd ${lm_bwd} \
        --recog_lm_weight ${lm_weight} \
        --recog_lm_usage ${lm_usage} \
        --recog_ctc_weight ${ctc_weight} \
        --recog_resolving_unk ${resolving_unk} \
        --recog_fwd_bwd_attention ${fwd_bwd_attention} \
        --recog_bwd_attention ${bwd_attention} \
        --recog_reverse_lm_rescoring ${reverse_lm_rescoring} \
        --recog_asr_state_carry_over ${asr_state_carry_over} \
        --recog_lm_state_carry_over ${lm_state_carry_over} \
        --recog_n_caches ${n_caches} \
        --recog_cache_theta_speech ${cache_theta_speech} \
        --recog_cache_lambda_speech ${cache_lambda_speech} \
        --recog_cache_theta_lm ${cache_theta_lm} \
        --recog_cache_lambda_lm ${cache_lambda_lm} \
        --recog_cache_type ${cache_type} \
        --recog_oracle ${oracle} \
        --recog_stdout ${stdout} || exit 1;

    # remove <unk>
    cat ${recog_dir}/ref.trn | sed 's:<unk>::g' > ${recog_dir}/ref.trn.filt
    cat ${recog_dir}/hyp.trn | sed 's:<unk>::g' > ${recog_dir}/hyp.trn.filt

    if [ ${metric} = 'edit_distance' ]; then
        echo ${set}
        sclite -r ${recog_dir}/ref.trn.filt trn -h ${recog_dir}/hyp.trn.filt trn -i rm -o all stdout > ${recog_dir}/result.txt
        grep -e Avg -e SPKR -m 2 ${recog_dir}/result.txt > ${recog_dir}/RESULTS
        cat ${recog_dir}/RESULTS
    fi
done
