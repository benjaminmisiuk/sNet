#' Train a perceptron matrix mapper
#' 
#' A shitty wrapper for [optim], which will evaluate optimization of weights and biases to fit one matrix to another using perceptrons.
#' 
#' @details This function optimizes weights and biases of perceptrons to map input matrix `x` to `y`. Perceptrons are densely connected to all inputs and outputs.
#' 
#' @param x Input matrix.
#' @param y Target matrix.
#' @param units Vector. Number of hidden densely connected perceptron units.
#' @param method Optimization method, most likely from [optim].
#' @param oprim_control List of control arguments to pass to [optim].
#' @param act Optional activation function to pass through after summing weights and biases of perceptrons.
#' @param ... Additional arguments to pass to the activation function.
#' 
#' @return A list with the [optim] output and other arguments specified in the call.
#' 
#' @note The length of `units` will determine the output length.
#' 
#' @export

ptron_train <- function(x, y, units = ncol(x), method = 'Nelder-Mead', optim_control = list(trace = 1, maxit = 5000), act = NULL, ...){
  opt <- optim(
    par = c(rep(1, ncol(x) * units), rep(0, units)),
    method = method,
    control = optim_control,
    fn = function(param = par){
      wt = matrix(param[1:(ncol(x) * units)], ncol = units)
      bias = param[(ncol(x) * units + 1):length(param)]
      out <- do.call(rbind, apply(x, 1, simplify = FALSE, FUN = function(z) colSums(z * wt) + bias))
      if(!is.null(act)) out <- do.call(rbind, apply(out, 1, simplify = FALSE, act))
      return(mean(as.matrix(y - out)^2))
    }
  )
  return(list(opt = opt, method = method, optim_control = optim_control, units = units, act = act))
}

#' Train an imperceptron matrix mapper
#' 
#' Another shitty wrapper for [optim]. This one optimizes the mapping between two matrices without any interaction between perceptrons until the 
#' final activation function, if specified.
#' 
#' @details Optimize weights and biases of non-dense perceptrons to map input matrix `x` to `y`. There are no dense hidden units.
#' 
#' @param x Input matrix.
#' @param y Target matrix.
#' @param method Optimization method, most likely from [optim].
#' @param oprim_control List of control arguments to pass to [optim].
#' @param act Optional activation function to pass through after summing weights and biases of perceptrons.
#' @param ... Additional arguments to pass to the activation function.
#' 
#' @return A list with the [optim] output and other arguments specified in the call.
#' 
#' @export

imptron_train <- function(x, y, method = 'Nelder-Mead', act = NULL, optim_control = list(trace = 1, maxit = 5000), ...){
  opt <- optim(
    par = c(rep(1, ncol(x)), rep(0, ncol(x))),
    method = method,
    control = optim_control,
    fn = function(param = par){
      wt = param[1:ncol(x)]
      bias = param[(ncol(x) + 1):length(param)]
      out <- do.call(rbind, apply(x, 1, simplify = FALSE, FUN = function(z) (z * wt) + bias))
      if(!is.null(act)) out <- do.call(rbind, apply(out, 1, simplify = FALSE, act))
      return(mean(as.matrix(y - out)^2))
    }
  )
  return(list(opt = opt, method = method, optim_control = optim_control, act = act))
}
