#Point a - 3b Meteo gives point forecasts for the temperature
#but probabilistic forecast for the rainfalls 

#Point b - Forecasting of the euro area from ECB website is probabilistic
#they do not give a number but a probability distribution
#on x axe there are the inflation intervals and on y 
#the probability assigned to each interval
#there are three surveys from 3 quarters to see how they evolve
#the distribution tends to be left skewed

install.packages("tidyverse")
install.packages("tidyr")
install.packages("ggplot2")
library(tidyverse)
library(dplyr)
library(ggplot2)

install.packages("CRAN")

data2024q3<-data %>% filter(quarter=="2024 Q3") 

sigmas <- function(m, mu, var) {
  delta <- (m - mu) * sqrt(pi/2)   # = sigma2 - sigma1
  c_term <- (1 - 2/pi) * delta^2 - var  # termine costante della quadratica
  
  discriminant <- delta^2 - 4 * c_term
  
  sigma1_1 <- (-delta + sqrt(discriminant)) / 2
  sigma1_2 <- (-delta - sqrt(discriminant)) / 2
  
  # sigma1 > 0
  sigma1 <- ifelse(sigma1_1 > 0, sigma1_1, sigma1_2)
  sigma2 <- delta + sigma1
  
  return(list(sigma1 = sigma1, sigma2 = sigma2))
}

test<- sigmas(m=data2024q3$mean1, mu=data2024q3$mode1, var=data2024q3$var1)
print(test)

density<- function(y,m,mu,var) {
  
  delta<-(m-mu)/((pi/2)^(-1/2))
  
  prod<- var-(1-2/pi)*(delta)^2
  
  sigma1<- (-delta+sqrt(delta^2+4*prod))/2

  sigma2<- delta+sigma1

  pdf<- ifelse(y<=mu, 
               (2/(sigma1+sigma2))*dnorm(y, mean=mu, sd=sigma1), 
               (2/(sigma1+sigma2))*dnorm(y, mean=mu, sd=sigma2))
  
  return(pdf)
  }



#needed because it is not a normal distribution
y_grid <- seq(-2, 8, length.out = 500)
obs <- data2024q3$inflation

# Colors
cols <- rainbow(7)

#plotting
plot(y_grid, density(y_grid, data2024q3$mean0, data2024q3$mode0, data2024q3$var0),
     type="l", col=cols[1], ylim=c(0,2.5), xlab="Inflation (%)", ylab="Density",
     main="Predictive densities for 2024 Q3")

for (h in 1:6) {
  m_h   <- data2024q3[[paste0("mean", h)]]
  mu_h  <- data2024q3[[paste0("mode", h)]]
  var_h <- data2024q3[[paste0("var",  h)]]
  lines(y_grid, density(y_grid, m_h, mu_h, var_h), col=cols[h+1])
}

abline(v=obs, col="black", lwd=2, lty=2)
legend("topright", legend=paste("h =", 0:6), col=cols, lty=1, cex=0.7)

#plotting now all observed inflation rates

ggplot(data, aes(x=quarter, y=inflation, group=1)) +
  geom_line() +
  labs(title="Inflation Rate 2005 Q3 - 2025 Q3",
       x="Quarter", y="Inflation (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=45, hjust=1))



