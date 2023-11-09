#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
# 
#-------------------------------------------------------
#
# This bash script installs the flores200 dataset, downloads laser2, and then
# performs xsim (multilingual similarity) evaluation with ratio margin


if [ -z ${LASER} ] ; then 
  echo "Please set the environment variable 'LASER'"
  exit
fi

pivot_lang="eng_Latn"
output_dir="${LASER}/tasks/xsim"
n_way=false
multilingual=false
while getopts o:l:mn flag
do
    case ${flag} in
        o) output_dir=${OPTARG};;
        l) pivot_lang=${OPTARG};;
        m) multilingual=true;;
        n) n_way=true;;
    esac
done

ddir="${LASER}/data"
cd $ddir  # move to data directory

if [ ! -d $ddir/flores200 ] ; then
    echo " - Downloading flores200..."
    wget --trust-server-names -q https://tinyurl.com/flores200dataset
    tar -xf flores200_dataset.tar.gz
    /bin/mv flores200_dataset flores200
    /bin/rm flores200_dataset.tar.gz
else
    echo " - flores200 already downloaded"
fi

mdir="${LASER}/models"
if [ ! -d ${mdir} ] ; then
  echo " - creating model directory: ${mdir}"
  mkdir -p ${mdir}
fi

function download {
    file=$1
    if [ -f ${mdir}/${file} ] ; then
        echo " - ${mdir}/$file already downloaded";
    else
        echo " - Downloading $s3/${file}";
        wget -q $s3/${file};
    fi 
}

cd $mdir  # move to model directory

# available encoders
s3="https://dl.fbaipublicfiles.com/nllb/laser"

if [ ! -f ${mdir}/laser2.pt ] ; then
    echo " - Downloading $s3/laser2.pt"
    wget --trust-server-names -q https://tinyurl.com/nllblaser2
else
    echo " - ${mdir}/laser2.pt already downloaded"
fi
download "laser2.spm"
download "laser2.cvocab"

corpus_part="devtest"
corpus="flores200"

# note: example evaluation script expects format: basedir/corpus/corpus_part/lang.corpus_part

langs_multi="eng_Latn fra_Latn deu_Latn spa_Latn rus_Cyrl arb_Arab jpn_Jpan kor_Hang hin_Deva swh_Latn"
langs_noisy="eng_Latn eng_abr1_Latn eng_abr2_Latn eng_abr3_Latn eng_fing_Latn eng_case_Latn eng_homo_Latn eng_cont_Latn eng_dysl_Latn eng_leet_Latn eng_spel_Latn eng_slng_Latn eng_week_Latn eng_spac_Latn"

if [ "$multilingual" = true ]; then
    langs=$langs_multi
    type="multi"
else
    langs=$langs_noisy
    type="noisy"
fi

langs=$(echo $langs | xargs) # trim trailing spaces
langs=${langs// /,} # replace spaces with commas

if [ "$n_way" = true ]; then
    output_file=$output_dir/${type}_m2m_error_matrix.csv

    echo " - calculating xsim (many-to-many)"
    python3 $LASER/source/eval.py                \
        --base-dir $ddir                         \
        --corpus $corpus                         \
        --corpus-part $corpus_part               \
        --margin ratio                           \
        --src-encoder   $LASER/models/laser2.pt  \
        --src-spm-model $LASER/models/laser2.spm \
        --src-langs $langs \
        --nway --verbose \
        --output-file $output_file
    echo "Results saved in $output_file"
else
    output_file=$output_dir/${pivot_lang}_${type}_o2m_error_matrix.csv
    
    echo " - calculating xsim (1-to-many)"
    python3 $LASER/source/eval.py                \
        --base-dir $ddir                         \
        --corpus $corpus                         \
        --corpus-part $corpus_part               \
        --margin ratio                           \
        --src-encoder   $LASER/models/laser2.pt  \
        --src-spm-model $LASER/models/laser2.spm \
        --src-langs $pivot_lang \
        --tgt-langs $langs \
        --verbose \
        --output-file $output_file
    echo "Results saved in $output_file"

    output_file=$output_dir/${pivot_lang}_${type}_m2o_error_matrix.csv

    echo " - calculating xsim (many-to-1)"
    python3 $LASER/source/eval.py                \
        --base-dir $ddir                         \
        --corpus $corpus                         \
        --corpus-part $corpus_part               \
        --margin ratio                           \
        --src-encoder   $LASER/models/laser2.pt  \
        --src-spm-model $LASER/models/laser2.spm \
        --src-langs $langs \
        --tgt-langs $pivot_lang \
        --verbose \
        --output-file $output_file
    echo "Results saved in $output_file"
fi
echo "Done..."