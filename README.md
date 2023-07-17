# sNet
Value mapping and optimization using simple perceptrons

This package facilitates the prediction of composition data values using arbitrary non-compositional models. 
The purpose is to avoid issues associated with transformation, zero values, data imputation, and other challenges when predicting compositional data challenges.

The package works by optimizing a simple auxiliary model on top of your primary model. 
This optimized model is similar to a simple neural network. 
It includes a series of weights and biases (like a regression), which map input values to the outputs.
The optimized model includes an activation function before the output. Using an appropriate function, predictions may be constrained to a compositional scale.

If the input and target output data values are the same, this may be conceptualized as a model that aims to predict a new set of values that are as close as possible to the input, with the new constraint of the activation function (e.g., that the predictions sum to 1).

Let's look at an example.

