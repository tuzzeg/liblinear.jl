#!/bin/sh

function train_predict() {
  file=$1
  train_args=$2
  train -s 0 "$train_args" $file.train.xy $file.model
  predict -b 1 $file.test.xy $file.model $file.res.xy
}

train_predict iris_1
train_predict sparse_1

train_predict iris_2 "-B 1"
