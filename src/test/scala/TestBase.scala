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

    def format_vec(v: DenseVector[Double]): String = {
        return "vec[" + v.data.map(x => "%.3f".format(x)).mkString(", ") + "]"
    }
}
