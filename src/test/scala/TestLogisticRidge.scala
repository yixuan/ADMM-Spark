package statr.stat598bd

import org.scalatest.FunSuite
import breeze.linalg._
import breeze.numerics._

class TestLogisticRidge extends TestBase {
    val f = "other/data.txt"
    val (x, y) = read_data(f)

    test("Vanilla logistic regression") {
        val mod = new LogisticRidge(x, y)
        mod.run()
        info(format_vec(mod.coef))
        info("# of iterations: " + mod.niter)
    }

    test("Vanilla logistic regression (native code)") {
        val mod = new LogisticRidgeNative(x, y)
        mod.run()
        info(format_vec(mod.coef))
        info("# of iterations: " + mod.niter)
    }

    test("Vanilla logistic regression using Newton's method") {
        val mod = new LogisticRidgeNewton(x, y)
        mod.run()
        info(format_vec(mod.coef))
        info("# of iterations: " + mod.niter)
    }

    test("Vanilla logistic regression using Newton's method (native code)") {
        val mod = new LogisticRidgeNewtonNative(x, y)
        mod.run()
        info(format_vec(mod.coef))
        info("# of iterations: " + mod.niter)
    }

    test("Shrinking beta to zero") {
        val mod = new LogisticRidge(x, y)
        mod.set_lambda(2.0)
        mod.run()
        info(format_vec(mod.coef))
        info("# of iterations: " + mod.niter)
    }

    test("Shrinking beta to zero (native code)") {
        val mod = new LogisticRidgeNative(x, y)
        mod.set_lambda(2.0)
        mod.run()
        info(format_vec(mod.coef))
        info("# of iterations: " + mod.niter)
    }

    test("Shrinking beta to zero using Newton's method") {
        val mod = new LogisticRidgeNewton(x, y)
        mod.set_lambda(2.0)
        mod.run()
        info(format_vec(mod.coef))
        info("# of iterations: " + mod.niter)
    }

    test("Shrinking beta to zero using Newton's method (native code)") {
        val mod = new LogisticRidgeNewtonNative(x, y)
        mod.set_lambda(2.0)
        mod.run()
        info(format_vec(mod.coef))
        info("# of iterations: " + mod.niter)
    }

    test("Shrinking beta to one") {
        val ones = DenseVector.ones[Double](x.cols)
        val mod = new LogisticRidge(x, y)
        mod.set_lambda(2.0)
        mod.set_v(ones)
        mod.run()
        info(format_vec(mod.coef))
        info("# of iterations: " + mod.niter)
    }

    test("Shrinking beta to one (native code)") {
        val ones = DenseVector.ones[Double](x.cols)
        val mod = new LogisticRidgeNative(x, y)
        mod.set_lambda(2.0)
        mod.set_v(ones)
        mod.run()
        info(format_vec(mod.coef))
        info("# of iterations: " + mod.niter)
    }

    test("Shrinking beta to one using Newton's method") {
        val ones = DenseVector.ones[Double](x.cols)
        val mod = new LogisticRidgeNewton(x, y)
        mod.set_lambda(2.0)
        mod.set_v(ones)
        mod.run()
        info(format_vec(mod.coef))
        info("# of iterations: " + mod.niter)
    }

    test("Shrinking beta to one using Newton's method (native code)") {
        val ones = DenseVector.ones[Double](x.cols)
        val mod = new LogisticRidgeNewtonNative(x, y)
        mod.set_lambda(2.0)
        mod.set_v(ones)
        mod.run()
        info(format_vec(mod.coef))
        info("# of iterations: " + mod.niter)
    }
}
