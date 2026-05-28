%This is a draft for the problemset 4%

install.packages("remotes")
remotes::install_github("aijordan/murphydiagram2")
set.seed(100)
n    <- 2000
mu   <- rnorm(n)
y    <- rnorm(n) + mu
tau  <- sample(c(-2, 2), prob = c(.5, .5), size = n, replace = TRUE)  # CHANGED
 
# --- Forecast CDFs (same as Problem 6) ----------------------------------------
F.perf  <- function(x) pnorm(x - mu)
F.clim  <- function(x) pnorm(x / sqrt(2))
F.unf   <- function(x) 0.5 * (pnorm(x - mu) + pnorm(x - mu - tau))
F.sign  <- function(x) pnorm(x + mu)
 
sd.under <- 3/4
sd.over  <- 5/4
pbias    <-  0.3
nbias    <- -0.3
 
F.under  <- function(x) pnorm((x - mu) / sd.under)
F.over   <- function(x) pnorm((x - mu) / sd.over)
F.pbias  <- function(x) pnorm(x - (mu + pbias))
F.nbias  <- function(x) pnorm(x - (mu + nbias))
 
# Collect all forecasts
forecasts <- list(
  Perfect        = F.perf,
  Climatological = F.clim,
  Unfocused      = F.unf,
  SignReversed   = F.sign,
  Overdispersed  = F.over,
  Underdispersed = F.under,
  Overprediction = F.pbias,
  Underprediction= F.nbias
)
 
forecast_names <- names(forecasts)
K <- length(forecasts)
 
# --- Install packages if needed -----------------------------------------------
# install.packages("murphydiagram")
# install.packages("remotes")
# remotes::install_github("aijordan/murphydiagram2")
 
library(murphydiagram)
library(murphydiagram2)
 
################################################################################
# FUNCTIONAL 1: Probability of the event {Y > 2}
# The eliciting functional is an exceedance probability: T(F) = 1 - F(2)
# The consistent scoring function family is the Brier score family (threshold t=2)
# Each forecaster reports: p_k = 1 - F_k(2)
################################################################################
 
# Extract the functional: P(Y > 2) from each forecast
prob_gt2 <- sapply(forecasts, function(F) 1 - F(2))
# prob_gt2 is a matrix of dimension n x K (one value per obs per forecast)
# since the CDFs are vectorised over mu/tau, each call returns a vector of length n
 
# Binary outcome: 1 if y > 2, 0 otherwise
z <- as.numeric(y > 2)
 
# Murphy diagram for functional 1 using murphydiag (all forecasts at once)
# The elementary scoring function for an exceedance probability at threshold t is:
#   S_t(p, y) = (1{y > t} - p)^2  -->  this is the Brier score at t=2
# murphydiag expects a matrix where columns are forecasts and rows are observations
 
cat("\n--- Functional 1: P(Y > 2) ---\n")
mat_prob <- prob_gt2  # n x K matrix
 
png("murphy_prob_gt2.png", width = 800, height = 500)
murphydiag(
  mat_prob,
  y       = z,
  type    = "prob",          # binary probability forecasts
  plot    = TRUE,
  col     = 1:K,
  lty     = 1:K,
  main    = "Murphy diagram: P(Y > 2)",
  xlab    = "Threshold t",
  ylab    = "Elementary score"
)
legend("topright", legend = forecast_names, col = 1:K, lty = 1:K, cex = 0.7)
dev.off()
 
# Area under Murphy curve for P(Y > 2)
auc_prob <- apply(mat_prob, 2, function(p) {
  # mean Brier score = mean (p - z)^2
  mean((p - z)^2)
})
cat("Area under Murphy curve (= mean Brier score) for P(Y > 2):\n")
print(round(auc_prob, 4))
 
 
################################################################################
# FUNCTIONAL 2: Mean
# The eliciting functional is the mean: T(F) = E_F[Y]
# For N(mu, sigma^2): mean = mu
# Consistent scoring functions: S_t(x, y) = (1{y <= t} - 1{x <= t})*(t - y)
#   which corresponds to the "mean" version in murphydiag
################################################################################
 
# Extract mean from each forecast
# Perfect:        E[Y|mu] = mu
# Climatological: E[Y]    = 0
# Unfocused:      E[Y|mu,tau] = 0.5*(mu + mu+tau) = mu + tau/2
# Sign-reversed:  E[Y|-mu] = -mu
# Overdispersed:  E[Y|mu] = mu  (mean unchanged)
# Underdispersed: E[Y|mu] = mu
# Overprediction: E[Y|mu+pbias] = mu + pbias
# Underprediction:E[Y|mu+nbias] = mu + nbias
 
means_mat <- cbind(
  Perfect         = mu,
  Climatological  = rep(0, n),
  Unfocused       = mu + tau / 2,
  SignReversed    = -mu,
  Overdispersed   = mu,
  Underdispersed  = mu,
  Overprediction  = mu + pbias,
  Underprediction = mu + nbias
)
 
cat("\n--- Functional 2: Mean ---\n")
 
png("murphy_mean.png", width = 800, height = 500)
murphydiag(
  means_mat,
  y    = y,
  type = "mean",
  plot = TRUE,
  col  = 1:K,
  lty  = 1:K,
  main = "Murphy diagram: Mean",
  xlab = "Threshold t",
  ylab = "Elementary score"
)
legend("topright", legend = forecast_names, col = 1:K, lty = 1:K, cex = 0.7)
dev.off()
 
# Area under Murphy curve for Mean (= mean squared error / CRPS up to constant)
auc_mean <- colMeans((means_mat - y)^2)
cat("Area under Murphy curve (= MSE) for Mean:\n")
print(round(auc_mean, 4))
 
 
################################################################################
# FUNCTIONAL 3: 90%-quantile
# The eliciting functional is the 0.9-quantile: T(F) = F^{-1}(0.9)
# Consistent scoring functions: pinball / asymmetric piecewise linear loss
################################################################################
 
# Extract 0.9-quantile from each forecast
# For N(mu, sigma^2): q_alpha = mu + sigma * qnorm(alpha)
alpha <- 0.9
 
# Mixture quantile (Unfocused): need numerical inversion
q_mixnorm <- function(p, mu_val, tau_val) {
  # CDF of 0.5*N(mu,1) + 0.5*N(mu+tau,1)
  cdf <- function(x) 0.5 * (pnorm(x - mu_val) + pnorm(x - mu_val - tau_val))
  uniroot(function(x) cdf(x) - p,
          interval = c(mu_val - 10, mu_val + 10))$root
}
 
q_unf <- mapply(q_mixnorm, p = alpha, mu_val = mu, tau_val = tau)
 
quant_mat <- cbind(
  Perfect         = mu + qnorm(alpha),
  Climatological  = sqrt(2) * qnorm(alpha),
  Unfocused       = q_unf,
  SignReversed    = -mu + qnorm(alpha),
  Overdispersed   = mu + sd.over  * qnorm(alpha),
  Underdispersed  = mu + sd.under * qnorm(alpha),
  Overprediction  = mu + pbias + qnorm(alpha),
  Underprediction = mu + nbias + qnorm(alpha)
)
 
cat("\n--- Functional 3: 90%-Quantile ---\n")
 
png("murphy_quantile90.png", width = 800, height = 500)
murphydiag(
  quant_mat,
  y     = y,
  type  = "quantile",
  level = alpha,
  plot  = TRUE,
  col   = 1:K,
  lty   = 1:K,
  main  = "Murphy diagram: 90%-Quantile",
  xlab  = "Threshold t",
  ylab  = "Elementary score"
)
legend("topright", legend = forecast_names, col = 1:K, lty = 1:K, cex = 0.7)
dev.off()
 
# Area under Murphy curve for 90%-quantile (= mean pinball loss)
pinball <- function(q, y, alpha) mean((y <= q) * (q - y) * (1 - alpha) +
                                      (y >  q) * (y - q) * alpha)
auc_quant <- apply(quant_mat, 2, pinball, y = y, alpha = alpha)
cat("Area under Murphy curve (= mean pinball loss) for 90%-quantile:\n")
print(round(auc_quant, 4))
 
 
################################################################################
# Summary table
################################################################################
 
summary_auc <- rbind(
  "P(Y>2) - Brier"    = auc_prob,
  "Mean - MSE"        = auc_mean,
  "Q90 - Pinball"     = auc_quant
)
 
cat("\n=== Area under Murphy curve (all functionals) ===\n")
print(round(summary_auc, 4))