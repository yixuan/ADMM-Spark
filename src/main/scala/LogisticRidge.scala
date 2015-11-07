package statr.stat598bd

import breeze.linalg._
import breeze.numerics._
import scala.util.control._

// Minimize
//     -loglik(beta) + lambda/2 * ||beta-v||^2
class LogisticRidge(val x: DenseMatrix[Double], val y: DenseVector[Double]) {
    // Dimension constants
    private val dim_n = x.rows
    private val dim_p = x.cols
    // Penalty parameters
    private var lambda: Double = -1.0
    private var v: DenseVector[Double] = DenseVector.zeros[Double](dim_p)
    // Parameters related to convergence
    private var max_iter: Int = 100
    private var eps_abs: Double = 1e-6
    private var eps_rel: Double = 1e-6
    // Variables to be returned
    private val bhat = DenseVector.zeros[Double](dim_p)
    private var iter = 0
    // Intermediate results that can be cached
    private val H0 = 0.25 * x.t * x
    private val solver = new Cholesky(dim_p)

    // pi(x, b) = 1 / (1 + exp(-x * b))
    private def pi(x: DenseMatrix[Double], b: DenseVector[Double]): DenseVector[Double] = {
        return 1.0 / (exp(- x * b) + 1.0)
    }

    def set_opts(max_iter: Int = 100, eps_abs: Double = 1e-6, eps_rel: Double = 1e-6) {
        this.max_iter = max_iter
        this.eps_abs = eps_abs
        this.eps_rel = eps_rel
    }

    def set_lambda(lambda: Double) {
        this.lambda = lambda

        // Hessian = X'WX + lambda * I
        // To simplify computation, use 0.25*I to replace W
        // 0.25*I >= W, in the sense that 0.25*I - W is p.d.
        val H = H0.copy
        for(i <- 0 until dim_p)
            H(i, i) = H(i, i) + lambda
        solver.compute(H)
    }

    def set_v(v: DenseVector[Double]) {
        this.v = v
    }

    def run() {
        if(lambda < 0)
            set_lambda(0.0)

        bhat := 0.0

        val loop = new Breaks
        loop.breakable {
            for(i <- 0 until max_iter) {
                val mu = pi(x, bhat)
                // Gradient = -X'(y-mu) + lambda*(beta-v)
                val grad = x.t * (mu - y) + lambda * (bhat - v)
                val delta = solver.solve(grad)
                bhat -= delta
                iter = i
                val r = norm(delta)
                if(r < eps_abs || r < eps_rel * norm(bhat)) {
                    loop.break
                }
            }
        }
    }

    def coef = bhat.copy
    def niter = iter
}
