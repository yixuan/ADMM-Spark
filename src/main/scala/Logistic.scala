package statr.stat598bd

import breeze.linalg._
import breeze.numerics._
import scala.util.control._

class Logistic(val x: DenseMatrix[Double], val y: DenseVector[Double]) {
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

    def set_opts(max_iter: Int = 100, eps_abs: Double = 1e-6, eps_rel: Double = 1e-6) {
        this.max_iter = max_iter
        this.eps_abs = eps_abs
        this.eps_rel = eps_rel
    }

    def run() {
        // Hessian = X'WX
        // To simplify computation, use 0.25*I to replace W
        // 0.25*I >= W, in the sense that 0.25*I - W is p.d.
        val H = x.t * x
        val solver = new Cholesky(H)

        bhat := 0.0

        // Intermediate results
        val mu = DenseVector.zeros[Double](dim_n)
        val w = DenseVector.zeros[Double](dim_n)
        val grad = DenseVector.zeros[Double](dim_p)
        val delta = DenseVector.zeros[Double](dim_p)

        val loop = new Breaks
        loop.breakable {
            for(i <- 0 until max_iter) {
                // println("\n*** Iter " + i + "\n")
                mu := pi(x, bhat)
                w := mu :* (1.0 - mu)

                grad := x.t * (y - mu)
                // println("grad =")
                // println(grad)

                // val delta = Hinv * grad
                solver.solve(grad, delta)
                // println("delta =")
                // println(delta)

                bhat += delta / max(w)
                // println("bhat =")
                // println(bhat)

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
