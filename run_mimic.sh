#! /bin/bash

res=result/mimic3/
for i in $(seq 0 4)  # to run all random cv splits in one time
do
  THEANO_FLAGS=device=gpu0 python hap.py data/mimic3/ remap.seqs remap.seqs remap ${res}HAP/ --p2c_file remap.p2c --sep_attention --L2 0 --n_epochs 50 --seed ${i} --embed_file pretrained_embedding.npz 
  THEANO_FLAGS=device=gpu0 python hap.py data/mimic3/ remap.seqs remap.seqs remap ${res}HAP_lv3/ --p2c_file remap.p2c --sep_attention --level 2 --L2 0 --n_epochs 50 --seed ${i} --embed_file pretrained_embedding.npz 
  THEANO_FLAGS=device=gpu0 python hap.py data/mimic3/ remap.seqs remap.seqs remap ${res}HAP_lv2/ --p2c_file remap.p2c --sep_attention --level 1 --L2 0 --n_epochs 50 --seed ${i} --embed_file pretrained_embedding.npz 

  THEANO_FLAGS=device=cpu python gram.py data/mimic3/ remap.seqs remap.seqs remap ${res}GRAM/ --embed_file pretrained_embedding.npz --L2 0 --seed ${i}
  THEANO_FLAGS=device=cpu python gram.py data/mimic3/ remap.seqs remap.seqs remap ${res}GRAM_lv3/ --level 2 --embed_file pretrained_embedding.npz --L2 0 --seed ${i}
  THEANO_FLAGS=device=cpu python gram.py data/mimic3/ remap.seqs remap.seqs remap ${res}GRAM_lv2/ --level 1 --embed_file pretrained_embedding.npz --L2 0 --seed ${i}

  THEANO_FLAGS=device=cpu python gram.py data/mimic3/ remap.seqs remap.seqs remap ${res}RNN/ --L2 0 --leaf --seed ${i}
  THEANO_FLAGS=device=cpu python gram.py data/mimic3/ remap.seqs remap.seqs remap ${res}RNN_plus/ --L2 0 --leaf --embed_file pretrained_embedding_leaf.npz --seed ${i}

  THEANO_FLAGS=device=cpu python gram.py data/mimic3/ remap.seqs remap.seqs remap ${res}Rollup/ --L2 0 --rollup --seed ${i}
  THEANO_FLAGS=device=cpu python gram.py data/mimic3/ remap.seqs remap.seqs remap ${res}Rollup_plus/ --L2 0 --rollup --seed ${i} --embed_file pretrained_embedding_leaf.npz
  
done

