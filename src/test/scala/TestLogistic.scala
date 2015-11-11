package statr.stat598bd

import org.scalatest.FunSuite
import breeze.linalg._
import breeze.numerics._

class TestLogistic extends TestBase {
    val f = "other/data.txt"
    val (x, y) = read_data(f)

    test("Vanilla logistic regression") {
        val mod = new Logistic(x, y)
        mod.run()
        info(format_vec(mod.coef))
        info("# of iterations: " + mod.niter)
    }

    test("Vanilla logistic regression using Newton's method ") {
        val mod = new LogisticNewton(x, y)
        mod.run()
        info(format_vec(mod.coef))
        info("# of iterations: " + mod.niter)
    }
}
