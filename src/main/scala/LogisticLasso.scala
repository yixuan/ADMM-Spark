package statr.stat598bd

import breeze.linalg._
import breeze.numerics._
import scala.util.control._

// Minimize
//     -loglik(beta) + lambda * ||beta||_1
class LogisticLasso(val x: DenseMatrix[Double], val y: DenseVector[Double])
      extends ADMML1(x.cols) {

    private val xsolver = new LogisticRidgeNative(x, y)
    xsolver.set_lambda(rho)
    xsolver.set_opts(100, 1e-3, 1e-3)

    override protected def rho_changed_action() {
        xsolver.set_lambda(rho)
    }

    protected def update_x() {
        val v = admm_z - admm_y / rho;
        xsolver.set_v(v)
        xsolver.run()
        admm_x := xsolver.coef
    }

    override protected def logging(iter: Int) {
        if(iter % 10 == 0) {
            val xb = x * admm_z.toDenseVector
            val ll = sum((y :* xb) - log(exp(xb) + 1.0))
            val penalty = lambda * sum(abs(admm_z))
            val obj = -ll + penalty
            println("Iteration #" + iter + ": obj = " + obj)
        }
    }
}
