package statr.stat598bd

import breeze.linalg._
import breeze.numerics._
import org.ejml.data._
import org.ejml.factory._

class Cholesky(val n: Int) {
    private val solver = LinearSolverFactory.chol(n)

    def this(A: DenseMatrix[Double]) = {
        this(A.cols)
        compute(A)
    }

    def compute(A: DenseMatrix[Double]) {
        // DenseMatrix (breeze) is column-major, DenseMatrix64F (EJML) is row-major,
        // but since x is assumed to be symmetric, the result would still be correct
        val A_wrapper = DenseMatrix64F.wrap(n, n, A.data)
        solver.setA(A_wrapper)
    }

    def solve(b: DenseVector[Double]): DenseVector[Double] = {
        val b_wrapper = DenseMatrix64F.wrap(n, 1, b.data)
        val res = DenseVector.zeros[Double](n)
        val res_wrapper = DenseMatrix64F.wrap(n, 1, res.data)
        solver.solve(b_wrapper, res_wrapper)
        return res
    }

    def solve(b: DenseVector[Double], x: DenseVector[Double]) {
        val b_wrapper = DenseMatrix64F.wrap(n, 1, b.data)
        val x_wrapper = DenseMatrix64F.wrap(n, 1, x.data)
        solver.solve(b_wrapper, x_wrapper)
    }
}
