module c

# include("../deps/deps.jl")
const liblinear="/usr/local/Cellar/liblinear/1.94/lib/liblinear"

immutable FeatureNode
  index::Cint
  value::Cdouble
end

immutable Problem
  l::Cint
  n::Cint
  y::Ptr{Cdouble}
  x::Ptr{Ptr{FeatureNode}}
  bias::Cdouble # < 0 if no bias term
end

# type of solver
#   for multi-class classification
#    0 -- L2-regularized logistic regression (primal)
#    1 -- L2-regularized L2-loss support vector classification (dual)
#    2 -- L2-regularized L2-loss support vector classification (primal)
#    3 -- L2-regularized L1-loss support vector classification (dual)
#    4 -- support vector classification by Crammer and Singer
#    5 -- L1-regularized L2-loss support vector classification
#    6 -- L1-regularized logistic regression
#    7 -- L2-regularized logistic regression (dual)
#   for regression
#   11 -- L2-regularized L2-loss support vector regression (primal)
#   12 -- L2-regularized L2-loss support vector regression (dual)
#   13 -- L2-regularized L1-loss support vector regression (dual)
const L2R_LR = convert(Cint, 0)
const L2R_L2LOSS_SVC_DUAL = convert(Cint, 1)
const L2R_L2LOSS_SVC = convert(Cint, 2)
const L2R_L1LOSS_SVC_DUAL = convert(Cint, 3)
const MCSVM_CS = convert(Cint, 4)
const L1R_L2LOSS_SVC = convert(Cint, 5)
const L1R_LR = convert(Cint, 6)
const L2R_LR_DUAL = convert(Cint, 7)
const L2R_L2LOSS_SVR = convert(Cint, 11)
const L2R_L2LOSS_SVR_DUAL = convert(Cint, 12)
const L2R_L1LOSS_SVR_DUAL = convert(Cint, 13)

function valid_solver(solver::Int)
  0 <= solver <= 7 || 11 <= solver <= 13
end

immutable Parameter
  solver_type::Cint

  # -e epsilon : set tolerance of termination criterion
  #   -s 0 and 2
  #     |f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,
  #     where f is the primal function and pos/neg are # of
  #     positive/negative data (default 0.01)
  #   -s 11
  #     |f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)
  #   -s 1, 3, 4, and 7
  #     Dual maximal violation <= eps; similar to libsvm (default 0.1)
  #   -s 5 and 6
  #     |f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,
  #     where f is the primal function (default 0.01)
  #   -s 12 and 13
  #     |f'(alpha)|_1 <= eps |f'(alpha0)|,
  #     where f is the dual function (default 0.1)
  eps::Cdouble            # stopping criteria
  C::Cdouble              # cost
  nr_weight::Cint
  weight_label::Ptr{Cint}
  weight::Ptr{Cdouble}
  p::Cdouble              # epsilon in loss function of SVR
end

immutable Model
  param::Parameter
  nr_class::Cint # number of classes
  nr_feature::Cint
  w::Ptr{Cdouble}
  label::Ptr{Cint} # label of each class
  bias::Cdouble
end

function train(problem::Ptr{Problem}, param::Ptr{Parameter})
  ccall((:train, liblinear), Ptr{Model}, (Ptr{Problem}, Ptr{Parameter}), problem, param)
end

function train(problem::Problem, param::Parameter)
  ccall((:train, liblinear), Ptr{Model}, (Ptr{Problem}, Ptr{Parameter}), &problem, &param)
end

function free_model_content(model::Ptr{Model})
  ccall((:free_model_content, liblinear), Void, (Ptr{Model},), model)
end

function free_and_destroy_model(model::Ptr{Ptr{Model}})
  ccall((:free_and_destroy_model, liblinear), Void, (Ptr{Ptr{Model}},), model)
end

function destroy_param(param::Ptr{Parameter});
  ccall((:destroy_param, liblinear), Void, (Ptr{Parameter},), param)
end

# Function callback for silent log
print_null(s::Ptr{Uint8}) = nothing
const print_null_c = cfunction(print_null, Void, (Ptr{Uint8},))

function set_silence(silent::Bool)
  if silent
    ccall((:set_print_string_function, liblinear), Void, (Ptr{Void},), print_null_c)
  else
    ccall((:set_print_string_function, liblinear), Void, (Ptr{Void},), C_NULL)
  end
end

end # module
