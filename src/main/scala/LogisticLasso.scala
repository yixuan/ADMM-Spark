package statr.stat598bd

import breeze.linalg._
import breeze.numerics._
import scala.util.control._

// Minimize
//     -loglik(beta) + lambda * ||beta||_1
class LogisticLasso(val x: DenseMatrix[Double], val y: DenseVector[Double]) {
    val dim_n = x.rows
    val dim_p = x.cols

    var max_iter: Int = 100
    var eps_abs: Double = 1e-6
    var eps_rel: Double = 1e-6
    var rho: Double = 1.0
    var lambda: Double = 0.0

    val admm_x = DenseVector.zeros[Double](dim_p)
    var admm_z = new VectorBuilder[Double](dim_p).toSparseVector
    val admm_y = DenseVector.zeros[Double](dim_p)
    var iter = 0

    // pi(x, b) = 1 / (1 + exp(-x * b))
    private def pi(x: DenseMatrix[Double], b: DenseVector[Double]): DenseVector[Double] = {
        return 1.0 / (exp(- x * b) + 1.0)
    }

    private def soft_shreshold(vec: DenseVector[Double], penalty: Double): SparseVector[Double] = {
        val builder = new VectorBuilder[Double](vec.size)
        for(ind <- 0 until vec.size) {
            val v = vec(ind)
            if(v > penalty) {
                builder.add(ind, v - penalty)
            } else if(v < -penalty) {
                builder.add(ind, v + penalty)
            }
        }
        return builder.toSparseVector(true, true)
    }

    def set_opts(max_iter: Int, eps_abs: Double, eps_rel: Double) {
        this.max_iter = max_iter
        this.eps_abs = eps_abs
        this.eps_rel = eps_rel
    }

    def set_lambda(lambda: Double) {
        this.lambda = lambda
    }

    def run() {
        val xsolver = new LogisticRidge(x, y)
        xsolver.set_lambda(rho)

        val loop = new Breaks
        loop.breakable {
            for(i <- 0 until max_iter) {
                val v = admm_z - admm_y / rho;
                xsolver.set_v(v)
                xsolver.run()
                admm_x := xsolver.coef

                admm_z = soft_shreshold(admm_x + admm_y / rho, lambda / rho)

                admm_y :+= rho * (admm_x - admm_z)

                iter = i
                println("z = ")
                println(admm_z.toDenseVector)
            }
        }
    }

    def coef = admm_z.copy
    def niter = iter
}
