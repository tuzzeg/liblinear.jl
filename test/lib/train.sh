#!/bin/sh

train -s 0 iris_1.train.xy iris_1.model
predict -b 1 iris_1.test.xy iris_1.model iris_1.res.xy
