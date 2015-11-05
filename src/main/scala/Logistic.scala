package statr.stat598bd

import breeze.linalg._
import breeze.numerics._
import scala.util.control._

class Logistic(val x: DenseMatrix[Double], val y: DenseVector[Double]) {
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
        val xx = x.t * x
        val Hinv = inv(-0.25 * xx)

        bhat := 0.0

        val i = 0
        val loop = new Breaks
        loop.breakable {
            for(i <- 0 until max_iter) {
                // println("\n*** Iter " + i + "\n")
                val mu = pi(x, bhat)

                val grad = x.t * (y - mu)
                // println("grad =")
                // println(grad)

                val delta = Hinv * grad
                // println("delta =")
                // println(delta)

                bhat -= delta
                // println("bhat =")
                // println(bhat)

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
