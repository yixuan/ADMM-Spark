package statr.stat598bd

import breeze.linalg._
import breeze.numerics._
import scala.util.control._

// Minimize
//     -loglik(beta) + lambda/2 * ||beta-v||^2
class LogisticRidgeNewton(val x: DenseMatrix[Double], val y: DenseVector[Double]) {
    // Dimension constants
    private val dim_n = x.rows
    private val dim_p = x.cols
    // Penalty parameters
    private var lambda: Double = 0.0
    private var v: DenseVector[Double] = DenseVector.zeros[Double](dim_p)
    // Parameters related to convergence
    private var max_iter: Int = 100
    private var eps_abs: Double = 1e-6
    private var eps_rel: Double = 1e-6
    // Variables to be returned
    private val bhat = DenseVector.zeros[Double](dim_p)
    private var iter = 0

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
    }

    def set_v(v: DenseVector[Double]) {
        this.v = v
    }

    def run() {
        bhat := 0.0

        val loop = new Breaks
        loop.breakable {
            for(i <- 0 until max_iter) {
                val mu = pi(x, bhat)
                val w = mu :* (1.0 - mu)
                // Gradient = -X'(y-mu) + lambda*(beta-v)
                val grad = x.t * (mu - y) + lambda * (bhat - v)
                // Hessian = X'WX + lambda * I
                val wx = x(::, *) :* w
                val H = x.t * wx
                for(i <- 0 until dim_p)
                    H(i, i) += lambda
                val solver = new Cholesky(H)

                val delta = solver.solve(grad)
                bhat -= delta
                iter = i
                val r = norm(delta)
                if(r < eps_abs * math.sqrt(dim_p) || r < eps_rel * norm(bhat)) {
                    loop.break
                }
            }
        }
    }

    def coef = bhat.copy
    def niter = iter
}
