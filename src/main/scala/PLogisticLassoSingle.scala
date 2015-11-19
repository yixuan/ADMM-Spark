// This class is mainly for debugging the parallel algorithm

package statr.stat598bd

import breeze.linalg._
import breeze.numerics._
import scala.util.control._

// Minimize
//     loss(A * x, b) + lambda * ||x||_1
// <=> \sum loss(A_i * x_i, b_i) + lambda * ||z||_1
//      s.t x_i - z = 0
class PLogisticLassoSingle(datx: Array[DenseMatrix[Double]],
                           daty: Array[DenseVector[Double]])
      extends PADMML1Single(datx, daty) {

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
        println("Iteration #" + iter + ": z = " + admm_z.toDenseVector)
        println("          #" + iter + ": eps_p = " + eps_primal + ", res_p = " + resid_primal)
        println("          #" + iter + ": eps_d = " + eps_dual + ", res_d = " + resid_dual)
    }
}
