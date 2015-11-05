package statr.stat598bd

import breeze.linalg._
import breeze.numerics._
import scala.util.control._

// Minimize
//     -loglik(beta) + lambda * ||beta||_1
class LogisticLasso(val x: DenseMatrix[Double], val y: DenseVector[Double])
      extends ADMML1(x.cols) {

    private val xsolver = new LogisticRidge(x, y)
    xsolver.set_lambda(rho)

    protected def update_x() {
        val v = admm_z - admm_y / rho;
        xsolver.set_v(v)
        xsolver.run()
        admm_x := xsolver.coef
    }
}
