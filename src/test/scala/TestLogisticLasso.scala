package statr.stat598bd

import org.scalatest.FunSuite
import breeze.linalg._
import breeze.numerics._

class TestLogisticLasso extends TestBase {

    test("Logistic lasso") {
        val f = "other/data.txt"
        val (x, y) = read_data(f)
        val mod = new LogisticLasso(x, y)
        mod.set_lambda(2.0)
        mod.run()
        info(format_vec(mod.coef.toDenseVector))
        info("# of iterations: " + mod.niter)
    }

    test("Logistic lasso (large)") {
        val f = "other/data_large.txt"
        val (x, y) = read_data(f)
        val mod = new LogisticLasso(x, y)
        mod.set_opts(1000, 1e-3, 1e-3, logs = true)
        mod.set_lambda(100.0)
        mod.run()
        info(format_vec(mod.coef.toDenseVector))
        info("# of iterations: " + mod.niter)
    }
}
