require("src/svmlight.jl")
require("src/liblinear.jl")

import StatsBase: fit, predict
import MLBase: roc, recall, precision, f1score
import liblinear: ClassificationParams, ClassificationModel, RegressionModel

function read(file; features=0)
  open(file, "r") do f
    x, y = read_svmlight(f)
    return x, convert(Array{Int}, y)
  end
end

function run(file_prefix, silent=false)
  train_file = "$file_prefix"
  test_file = "$file_prefix.t"

  t0 = time()
  x_train, y_train = read(train_file)
  r_train, c_train = size(x_train)
  x_test, y_test = read(test_file; features=c_train)
  t1 = time()

  params = ClassificationParams((-1, 1); eps=0.01)
  model = fit(ClassificationModel{Int}, x_train, y_train, params)
  t_train = time()

  y1 = predict(model, x_test)
  t_test = time()

  if !silent
    rc = roc(y_test, y1)
    println("julia: train=$(size(x_train)) test=$(size(x_test))")
    @printf "  precision=%.5f recall=%.5f f1=%.5f\n" precision(rc) recall(rc) f1score(rc)
    @printf "  read=%.5f train=%.5f test=%.5f\n" t1-t0 t_train-t1 t_test-t_train
  end
end

file_prefix = ARGS[1]

# warm up
run(file_prefix, true)

# measure
run(file_prefix, false)
