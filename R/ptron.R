#' Perceptron matrix
#' 
#' Calculate the values at a layer of perceptrons.
#' 
#' @details This function multiplies each row of a matrix of input values by a matrix of weights, adds a vector of biases, and optionally passes the output through
#' an activation function (such as found in [activations]). These parameters can be taken from the output of [ptron_train].
#' 
#' @param x Matrix of input values.
#' @param units Vector. Number of hidden densely connected perceptron units.
#' @param act Optional activation function to pass through after summing weights and bias. See functions included in [activations] or provide your own.
#' @param param Vector of weights and biases to be optimized (e.g., using [optim]). The first `ncol(mat) * units` are weights, and the remaining
#' are biases for each `unit`. This is normally tkaen from the output of [ptron_train].
#' @param ... Additional arguments to pass to the activation function.
#' 
#' @return A matrix of dimensions `nrow(x)` by `length(units)`.
#' 
#' @export
#' 

ptron <- function(x, units = ncol(x), act = NULL, param = c(rep(1, ncol(x) * units), rep(0, units)), ...){
  wt = matrix(param[1:(ncol(x) * units)], ncol = units)
  bias = param[(ncol(x) * units + 1):length(param)]
  
  out <- do.call(rbind, apply(x, 1, simplify = FALSE, FUN = function(z) colSums(z * wt) + bias))
  if(!is.null(act)) out <- do.call(rbind, apply(out, 1, simplify = FALSE, act))
  
  return(out)
}

#' Imperceptron matrix
#' 
#' Calculate values at a layer of perceptrons that match dimensions of the input matrix, with no interaction.
#' 
#' @details This function multiplies each row of an input matrix by a vector of weights, plus a vector of biases, and optionally passes the output through
#' an activation function (such as found in [activations]). These parameters can be taken from the output of [imptron_train].
#' 
#' @param x Matrix of input values.
#' @param act Optional activation function to pass through after summing weights and bias. See functions included in [activations] or provide your own.
#' @param param Vector of weights and biases to be optimized (e.g., using [optim]). The first `ncol(x)` are weights, and the remaining
#' are biases. This is normally taken from the output of [imptron_train].
#' @param ... Additional arguments to pass to the activation function.
#' 
#' @return Matrix of same dimensions as `x`.
#' 
#' @export

imptron <- function(x, act = NULL, param = c(rep(1, ncol(x)), rep(0, ncol(x))), ...){
  wt = param[1:ncol(x)]
  bias = param[(ncol(x) + 1):length(param)]
  
  out <- do.call(rbind, apply(x, 1, simplify = FALSE, FUN = function(z) (z * wt) + bias))
  if(!is.null(act)) out <- do.call(rbind, apply(out, 1, simplify = FALSE, act))
  
  return(out)
}