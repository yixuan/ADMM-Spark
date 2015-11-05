package statr.stat598bd

import breeze.linalg._
import breeze.numerics._
import scala.util.control._

// Minimize
//     -loglik(beta) + lambda/2 * ||beta-v||^2
class LogisticRidge(val x: DenseMatrix[Double], val y: DenseVector[Double],
                    val lambda: Double, val v: DenseVector[Double]) {
    var max_iter: Int = 100
    var eps_abs: Double = 1e-6
    var eps_rel: Double = 1e-6

    val dim_n = x.rows
    val dim_p = x.cols

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

    def run() {
        // Hessian = X'WX + lambda * I
        // To simplify computation, use 0.25*I to replace W
        // 0.25*I >= W, in the sense that 0.25*I - W is p.d.
        val H = 0.25 * x.t * x + lambda * DenseMatrix.eye[Double](dim_p)
        val Hinv = inv(H)

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
