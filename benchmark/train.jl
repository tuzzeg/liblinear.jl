require("src/svmlight.jl")
require("src/liblinear.jl")

import StatsBase: fit, predict
import MLBase: roc, recall, precision, f1score
import liblinear: ClassificationParams, ClassificationModel, RegressionModel


function read(file)
  open(ARGS[1], "r") do f
    x, y = read_svmlight(f)
    return x, convert(Array{Int}, y)
  end
end

file_prefix = ARGS[1]
train_file = "$file_prefix.train"
test_file = "$file_prefix.test"
t0 = time()
x_train, y_train = read(train_file)
x_test, y_test = read(test_file)
t1 = time()

params = ClassificationParams((-1, 1); eps=0.01)
model = fit(ClassificationModel{Int}, x_train, y_train, params)
t_train = time()

y1 = predict(model, x_test)
t_test = time()

rc = roc(y_test, y1)
@printf "julia: precision=%.5f recall=%.5f f1=%.5f\n" precision(rc) recall(rc) f1score(rc)
@printf "julia: read=%.3f train=%.3f test=%.3f" t1-t0 t_train-t1 t_test-t_train
