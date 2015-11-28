package statr.stat598bd

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import breeze.linalg._
import breeze.numerics._
import scala.util.control._

// Minimize
//     loss(A * x, b) + lambda * ||x||_1
// <=> \sum loss(A_i * x_i, b_i) + lambda * ||z||_1
//      s.t x_i - z = 0
class PLogisticLasso(datx: RDD[DenseMatrix[Double]],
                     daty: RDD[DenseVector[Double]],
                     sc: SparkContext)
      extends PADMML1(datx, daty, sc) {

    protected def update_x(x: DenseMatrix[Double], y: DenseVector[Double],
                           rho: Double, v: DenseVector[Double]): DenseVector[Double] = {
        val xsolver = new LogisticRidgeNative(x, y)
        xsolver.set_lambda(rho)
        xsolver.set_opts(100, 1e-3, 1e-3)
        xsolver.set_v(v)
        xsolver.run()
        return xsolver.coef
    }
    override protected def logging(iter: Int) {
        // Print header
        if (iter == 0) {
            println("=" * 80)
            println("%-7s%-15s%-15s%-15s%-15s%-15s".format("iter", "eps_primal", "resid_primal",
                                                           "eps_dual", "resid_dual", "rho"))
            println("-" * 80)
        }
        println("%-7d%-15f%-15f%-15f%-15f%-15f".format(iter, eps_primal, resid_primal,
                                                       eps_dual, resid_dual, rho))
    }
}
