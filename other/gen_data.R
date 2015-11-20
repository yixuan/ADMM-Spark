library(glmnet)
set.seed(123)
n = 100
p = 5

px = function(x, b) 1 / (1 + exp(-c(x %*% b)))

x = matrix(rnorm(n * p), n)
b = rnorm(p)
y = rbinom(n, 1, px(x, b))

m1 = glm(y ~ x + 0, family = binomial())
coef(m1)

m2 = glmnet(x, y, family = "binomial", standardize = FALSE, intercept = FALSE, alpha = 0)
coef(m2, 2 / n, exact = TRUE)

m3 = glmnet(x, y, family = "binomial", standardize = FALSE, intercept = FALSE)
coef(m3, 2 / n, exact = TRUE)

write.table(data.frame(y = y, x = x), "data.txt", col.names = FALSE, row.names = FALSE, sep = " ")



set.seed(123)
n = 10000
p = 200
x = matrix(rnorm(n * p), n)
b = rnorm(p)
y = rbinom(n, 1, px(x, b))
write.table(data.frame(y = y, x = x), "data_large.txt", col.names = FALSE, row.names = FALSE, sep = " ")
