# sNet
### Value mapping and optimization using simple perceptrons

This package facilitates the prediction of composition data values using arbitrary non-compositional models. 
The purpose is to avoid issues associated with transformation, zero values, data imputation, and other challenges when predicting compositional data.

The package works by optimizing a simple auxiliary model on top of raw multivariate (potentially compositional) prediction values. 
This optimized model is similar to a simple neural network. 
It includes a series of weights and biases (like a regression), which map input values to the outputs.
The optimized model includes an activation function before the output. Using an appropriate function, predictions may be constrained to a compositional scale.

If the input and target output data values are the same, this may be conceptualized as a model that aims to predict a new set of values that are as close as possible to the input, with the new constraint of the activation function (e.g., that the predictions sum to 1).

Let's look at an example.

```
#install or load the package
remotes::install_github('benjaminmisiuk/sNet')
library(sNet)

#load example data
data(gs)
head(gs)
```

There are four compositional measures of sediment grain size, followed by a number of different environmental measurements. We can build a model predicting each grain size parameter using the environmental data.

```
#model grain size composition using random forest
library(randomForest)

rf_clay <- randomForest(clay ~ . - silt - sand - gravel, gs)
rf_silt <- randomForest(silt ~ . - clay - sand - gravel, gs)
rf_sand <- randomForest(sand ~ . - clay - silt - gravel, gs)
rf_gravel <- randomForest(gravel ~ . - clay - silt - sand, gs)

#predict each grain size fraction
rf_pred <- data.frame(
  clay = predict(rf_clay),
  silt = predict(rf_silt),
  sand = predict(rf_sand),
  gravel = predict(rf_gravel)
)
```

By definition, compositional data must sum to 100% (or 1). This is the problem with modelling raw compositional data.

```
#predictions do not sum to 1
apply(rf_pred, 1, sum)
```

```
#     1         2         3         4         5         6         7         8         9        10        11        12        13        14        15        16 
#1.3316548 1.2342476 1.2376138 1.0717491 1.2171812 1.0556172 1.0839765 1.2380406 1.2084617 1.0749508 1.1345669 1.1639950 1.0869840 1.1178892 1.2661299 1.1174319 
#       17        18        19        20        21        22        23        24        25        26        27        28        29        30        31        32 
#1.1078956 1.2320431 1.0616280 1.0381852 1.1087825 1.3107085 1.0897432 1.0403853 0.9110006 1.1641133 0.8953935 1.0558356 1.0855090 0.9049162 1.0815615 1.1488688 
#...
```

We can use sNet to optimize the predictions for a compositional distribution by choosing an appropriate activation function. For compositional data, this could be `relu_norm`, `softmax`, or many others. We first must train the optimizer to map inputs to outputs. Provide both an input and target matrix (or columns of a data frame). For our example, this is just the prediction matrix for both input and output (we want to map the predictions to a compositional scale).

```
#fit an optimizer to the model predictions
imp <- imptron_train(rf_pred, rf_pred, act = relu_norm)
```

By default, the function uses "Nelder-Mead" optimization as implemented by [optim](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/optim.html) from the [stats](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/00Index.html) package. You can use other optimizers as well.

```
#Broyden, Fletcher, Goldfarb and Shanno (BFGS) optimization
imp <- imptron_train(rf_pred, rf_pred, method = 'BFGS', act = relu_norm)
```

Next, we use the function `imptron` to predict the new optimized compositional values, which sum to 1. `act` is the activation function from `imptron_train()` above, and `param` are the optimized perceptron parameters.

```
imp_pred <- imptron(rf_pred, act = imp$act, param = imp$opt$par)

#predictions sum to 1
apply(imp_pred, 1, sum)
```

```
#1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40 
#  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1 
# 41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80 
#  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1 
#...
```

We can compare the accuracy of the optimized predictions to the raw ones.

```
#create a function to calculate variance explained
ve <- function(ref, pred){
  1 - (sum((ref - pred)^2) / sum((ref - mean(ref))^2))
}
```

```
#accuracy of raw rf predictions
mean(
  c(
    ve(gs$clay, rf_pred$clay),
    ve(gs$silt, rf_pred$silt),
    ve(gs$sand, rf_pred$sand),
    ve(gs$gravel, rf_pred$gravel)
  )
)
```

```
#[1] 0.5241498
```

```
#accuracy of optimized predictions
mean(
  c(
    ve(gs$clay, imp_pred[ ,1]),
    ve(gs$silt, imp_pred[ ,2]),
    ve(gs$sand, imp_pred[ ,3]),
    ve(gs$gravel, imp_pred[ ,4])
  )
)
```

```
#[1] 0.5236641
```
