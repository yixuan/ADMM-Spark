package statr.stat598bd

import org.scalatest.FunSuite
import scala.io.Source
import breeze.linalg._
import breeze.numerics._

abstract class TestBase extends FunSuite {
    def read_data(f: String): (DenseMatrix[Double], DenseVector[Double]) = {
        var source = Source.fromFile(f)
        val lines = source.getLines()
        val first = lines.take(1).toArray
        val n = lines.length + 1
        val p = first(0).split(' ').length - 1
        source.close()

        val x = DenseMatrix.zeros[Double](n, p)
        val y = DenseVector.zeros[Double](n)

        source = Source.fromFile(f)
        var i = 0
        for(line <- source.getLines()) {
            val l = line.split(' ')
            y(i) = l(0).toDouble
            x(i, ::) := (new DenseVector(l.drop(1).map(x => x.toDouble))).t
            i += 1
        }
        source.close()

        return (x, y)
    }

    def read_data_parts(f: String): (Array[DenseMatrix[Double]], Array[DenseVector[Double]]) = {
        val (xall, yall) = read_data(f)
        val n = xall.rows
        val x1 = xall(0 until (n / 2), ::).copy
        val x2 = xall((n / 2) until n, ::).copy
        val y1 = yall(0 until (n / 2)).copy
        val y2 = yall((n / 2) until n).copy

        return (Array(x1, x2), Array(y1, y2))
    }

    def format_vec(v: DenseVector[Double]): String = {
        return "vec[" + v.data.map(x => "%.3f".format(x)).mkString(", ") + "]"
    }
}
