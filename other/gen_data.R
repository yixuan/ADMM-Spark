set.seed(123)
n = 100
p = 5

px = function(x, b) 1 / (1 + exp(-c(x %*% b)))

x = matrix(rnorm(n * p), n)
b = rnorm(p)
y = rbinom(n, 1, px(x, b))

glm(y ~ x + 0, family = binomial())

write.table(data.frame(y = y, x = x), "data.txt", col.names = FALSE, row.names = FALSE, sep = " ")
