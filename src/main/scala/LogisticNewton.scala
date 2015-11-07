package statr.stat598bd

import breeze.linalg._
import breeze.numerics._
import scala.util.control._

class LogisticNewton(val x: DenseMatrix[Double], val y: DenseVector[Double]) {
    // Dimension constants
    private val dim_n = x.rows
    private val dim_p = x.cols
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

    def set_opts(max_iter: Int, eps_abs: Double, eps_rel: Double) {
        this.max_iter = max_iter
        this.eps_abs = eps_abs
        this.eps_rel = eps_rel
    }

    def run() {

        bhat := 0.0

        // Intermediate results
        val delta = DenseVector.zeros[Double](dim_p)

        val loop = new Breaks
        loop.breakable {
            for(i <- 0 until max_iter) {
                val mu = pi(x, bhat)
                val w = sqrt(mu :* (1.0 - mu))
                val grad = x.t * (y - mu)

                val wx = x(::, *) :* w
                val solver = new Cholesky(wx.t * wx)
                solver.solve(grad, delta)

                bhat += delta

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
