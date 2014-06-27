using Base.Test

import StatsBase: fit, predict

require("src/liblinear.jl")
import liblinear: ClassificationParams, ClassificationModel, RegressionModel

x_train = sparse([
  0 0 1 0 1;
  0 1 1 0 0;
  1 0 0 0 1;
  0 0 1 0 0;
  0 1 0 1 0;
  1 1 0 0 1
])
y_train = [0, 0, 0, 1, 1, 1]

x_test = sparse([
  1 0 0 0 1;
  0 0 1 0 1;
  0 1 0 0 0;
  0 0 1 1 0
])
y_test = [0, 0, 1, 1]

function test_regression()
  params = ClassificationParams((0, 1); eps=0.01)

  model = fit(RegressionModel, x_train, y_train, params)
  @test_approx_eq_eps [-0.0487582,-0.31881,0.290554,-0.341164,0.306295] model.weights 1e-4

  y1 = predict(model, x_test)
  @test_approx_eq_eps [0.564031,0.644935,0.420966,0.4873] y1 1e-4
end

function test_regression_compare_with_dense()
  params = ClassificationParams((0, 1); eps=0.01)

  m_sparse = fit(RegressionModel, x_train, y_train, params)
  m_dense = fit(RegressionModel, full(x_train), y_train, params)
  @test_approx_eq_eps m_sparse.weights m_dense.weights 1e-4

  y1_sparse = predict(m_sparse, x_test)
  y1_dense = predict(m_dense, full(x_test))
  @test_approx_eq_eps y1_sparse y1_dense 1e-4
end

function test_regression_with_bias()
  params = ClassificationParams((0, 1); eps=0.01, bias=1.0)

  m_sparse = fit(RegressionModel, x_train, y_train, params)
  m_dense = fit(RegressionModel, full(x_train), y_train, params)
  @test_approx_eq_eps m_sparse.weights m_dense.weights 1e-4

  y1_sparse = predict(m_sparse, x_test)
  y1_dense = predict(m_dense, full(x_test))
  @test_approx_eq_eps y1_sparse y1_dense 1e-4
end

test_regression()
test_regression_compare_with_dense()

test_regression_with_bias()
