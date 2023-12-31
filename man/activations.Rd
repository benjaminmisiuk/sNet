% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/activations.R
\name{activations}
\alias{activations}
\alias{linear_norm}
\alias{softmax}
\alias{relu}
\alias{leaky_relu}
\alias{prelu}
\alias{tanh}
\alias{sigmoid}
\alias{relu_norm}
\title{Activations}
\usage{
linear_norm(z)

softmax(z)

relu(z)

leaky_relu(z, a = 0.01)

prelu(z, a)

tanh(z)

sigmoid(z)

relu_norm(z)
}
\arguments{
\item{z}{Input vector.}

\item{a}{Value of alpha for ReLU-like functions.}
}
\value{
A vector of \code{length(z)}.
}
\description{
Several common activation functions used for neural networks.
}
\details{
These activations all take a vector as input and return a transformed vector of same length.
}
