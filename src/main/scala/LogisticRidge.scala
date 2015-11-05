package statr.stat598bd

import breeze.linalg._
import breeze.numerics._
import scala.util.control._

// Minimize
//     -loglik(beta) + lambda/2 * ||beta-v||^2
class LogisticRidge(val x: DenseMatrix[Double], val y: DenseVector[Double]) {
    val dim_n = x.rows
    val dim_p = x.cols

    val xtx = 0.25 * x.t * x
    val Hinv = inv(xtx)

    var max_iter: Int = 100
    var eps_abs: Double = 1e-6
    var eps_rel: Double = 1e-6

    var lambda: Double = 0.0
    var v: DenseVector[Double] = DenseVector.zeros[Double](dim_p)

    val bhat = DenseVector.zeros[Double](dim_p)
    var iter = 0

    // pi(x, b) = 1 / (1 + exp(-x * b))
    private def pi(x: DenseMatrix[Double], b: DenseVector[Double]): DenseVector[Double] = {
        return 1.0 / (exp(- x * b) + 1.0)
    }

    def set_opts(max_iter: Int, eps_abs: Double, eps_rel: Double) {
        this.max_iter = max_iter
        this.eps_abs = eps_abs
        this.eps_rel = eps_rel
    }

    def set_lambda(lambda: Double) {
        this.lambda = lambda

        // Hessian = X'WX + lambda * I
        // To simplify computation, use 0.25*I to replace W
        // 0.25*I >= W, in the sense that 0.25*I - W is p.d.
        Hinv := inv(xtx + lambda * DenseMatrix.eye[Double](dim_p))
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
                // Gradient = -X'(y-mu) + lambda*(beta-v)
                val grad = x.t * (mu - y) + lambda * (bhat - v)
                val delta = Hinv * grad
                bhat -= delta
                iter = i
                val r = norm(delta)
                if(r < eps_abs || r < eps_rel * norm(bhat)) {
                    loop.break
                }
            }
        }
    }

    def coef = bhat
    def niter = iter
}
