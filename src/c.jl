module c

include("../deps/deps.jl")

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

# solver_type
const L2R_LR Cint = convert(Cint, 0)
const L2R_L2LOSS_SVC_DUAL Cint = convert(Cint, 1)
const L2R_L2LOSS_SVC Cint = convert(Cint, 2)
const L2R_L1LOSS_SVC_DUAL Cint = convert(Cint, 3)
const MCSVM_CS Cint = convert(Cint, 4)
const L1R_L2LOSS_SVC Cint = convert(Cint, 5)
const L1R_LR Cint = convert(Cint, 6)
const L2R_LR_DUAL Cint = convert(Cint, 7)
const L2R_L2LOSS_SVR Cint = convert(Cint, 11)
const L2R_L2LOSS_SVR_DUAL Cint = convert(Cint, 12)
const L2R_L1LOSS_SVR_DUAL Cint = convert(Cint, 13)

immutable Parameter
  solver_type::Cint

# these are for training only
  eps::Cdouble # stopping criteria
  C::Cdouble
  nr_weight::Cint
  weight_label::Ptr{Cint}
  weight::Ptr{Cdouble}
  double p::Cdouble
end

immutable Model
  param::Parameter
  nr_class::Cint # number of classes
  nr_feature::Cint
  w::Ptr{Cdouble}
  label::Ptr{Cint} # label of each class
  bias::Ptr{Cdouble}
end

function train(problem::Ptr{Problem}, param::Ptr{Parameter})
  ccall((:train, liblinear), Ptr{Model}, (Ptr{Problem}, Ptr{Parameter}), problem, param)
end

function free_model_content(model::Ptr{Model})
  ccall((:free_model_content, liblinear), Void, (Ptr{Model}), model)
end

function free_and_destroy_model(model::Ptr{Ptr{Model}})
  ccall((:free_and_destroy_model, liblinear), Void, (Ptr{Ptr{Model}},), model)
end

function destroy_param(param::Ptr{Parameter});
  ccall((:destroy_param, liblinear), Void, (Ptr{Parameter},), param)
end

end # module
