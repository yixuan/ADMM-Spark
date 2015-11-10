package statr.stat598bd

import com.github.fommil.jni.JniLoader
import breeze.linalg._

object LogisticRidgeNewtonNative {
    JniLoader.load("logistic_ridge.so")
}

// Minimize
//     -loglik(beta) + lambda/2 * ||beta-v||^2
class LogisticRidgeNewtonNative(val x: DenseMatrix[Double], val y: DenseVector[Double]) {
    // Trigger static block
    LogisticRidgeNewtonNative

    // Dimension constants
    private val dim_n = x.rows
    private val dim_p = x.cols
    // Penalty parameters
    private var lambda: Double = 0.0
    private var v: DenseVector[Double] = DenseVector.zeros[Double](dim_p)
    // Parameters related to convergence
    private var max_iter: Int = 100
    private var eps_abs: Double = 1e-6
    private var eps_rel: Double = 1e-6
    // Variables to be returned
    private val bhat = DenseVector.zeros[Double](dim_p)
    private var iter = 0

    def set_opts(max_iter: Int = 100, eps_abs: Double = 1e-6, eps_rel: Double = 1e-6) {
        this.max_iter = max_iter
        this.eps_abs = eps_abs
        this.eps_rel = eps_rel
    }

    def set_lambda(lambda: Double) {
        this.lambda = lambda
    }

    def set_v(v: DenseVector[Double]) {
        this.v = v
    }

    @native def logistic_ridge_newton(x: Array[Double], n: Int, p: Int, y: Array[Double],
                                      lambda: Double, v: Array[Double],
                                      max_iter: Int, eps_abs: Double, eps_rel: Double,
                                      niter: Array[Int]): Array[Double]

    def run() {
        val iter_arg = new Array[Int](1)
        val res = logistic_ridge_newton(x.data, dim_n, dim_p, y.data,
                                        lambda, v.data,
                                        max_iter, eps_abs, eps_rel, iter_arg)
        bhat := DenseVector(res)
        iter = iter_arg(0)
    }

    def coef = bhat.copy
    def niter = iter
}
