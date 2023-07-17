#' Activations
#' 
#' Several common activation functions used for neural networks.
#' 
#' @name activations
#' @rdname activations
#' 
#' @details These activations all take a vector as input and return a transformed vector of same length.
#' 
#' @param z Input vector.
#' @param a Value of alpha for ReLU-like functions.
#' 
#' @return A vector of `length(z)`.
#' 
NULL

#' @rdname activations
#' @export
#' 
normalize <- function(z){
  z / sum(z)
}

#' @rdname activations
#' @export
#'
softmax <- function(z){
  exp(z) / (sum(exp(z)))
}

#' @rdname activations
#' @export
#'
relu <- function(x){
  sapply(z, FUN=function(z) max(0, z))
}

#' @rdname activations
#' @export
#'
leaky_relu <- function(z, a = 0.01){
  sapply(z, FUN=function(zz) max(zz, a * zz))
}

#' @rdname activations
#' @export
#'
prelu <- function(z, a){
  sapply(z, FUN=function(zz) max(zz, zz * a))
}

#' @rdname activations
#' @export
#'
tanh <- function(z){
  (exp(z) - exp(-z)) / (exp(z) + exp(-z))
}

#' @rdname activations
#' @export
#'
sigmoid <- function(z){
  1 / (1 + exp(-1 * z))
}

##' @rdname activations
#' @export
#'
relu_norm <- function(z){
  z_i <- sapply(z, FUN=function(zz) max(0, zz))
  z_i / sum(z_i)
}