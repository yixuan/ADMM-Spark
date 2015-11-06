package statr.stat598bd

import breeze.linalg._
import breeze.numerics._
import org.ejml.data._
import org.ejml.factory._

class Cholesky(val A: DenseMatrix[Double]) {
    // DenseMatrix (breeze) is column-major, DenseMatrix64F (EJML) is row-major,
    // but since x is assumed to be symmetric, the result would be correct
    private val A_wrapper = DenseMatrix64F.wrap(A.rows, A.cols, A.data)

    private val solver = LinearSolverFactory.chol(A.cols)
    solver.setA(A_wrapper)

    def solve(b: DenseVector[Double]): DenseVector[Double] = {
        val b_wrapper = DenseMatrix64F.wrap(b.size, 1, b.data)
        val res = DenseVector.zeros[Double](b.size)
        val res_wrapper = DenseMatrix64F.wrap(res.size, 1, res.data)
        solver.solve(b_wrapper, res_wrapper)
        return res
    }

    def solve(b: DenseVector[Double], x: DenseVector[Double]) {
        val b_wrapper = DenseMatrix64F.wrap(b.size, 1, b.data)
        val x_wrapper = DenseMatrix64F.wrap(x.size, 1, x.data)
        solver.solve(b_wrapper, x_wrapper)
    }
}
