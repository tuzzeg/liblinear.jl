# TODO

## Cases:
+ Classification/Regression
+ 2 Class/Multiclass
+ Bias/no bias
+ dense/sparse

## Implement standard methods
+ StatBase.fit(::Type{Model}, x, y, params) :: Model
+ StatBase.predict(model::Model, x) -> x'

## Benchmarks
- benchmark/smoke test
  + train/test using dataset
  + calculate precision/recall numbers
  - confusion matrix
