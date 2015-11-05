package statr.stat598bd

import scala.io.Source
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._

object Demo {
    private def pi(x: DenseMatrix[Double], b: DenseVector[Double]): DenseVector[Double] = {
        return 1.0 / (exp(- x * b) + 1.0)
    }

    private def read_data(f: String): (DenseMatrix[Double], DenseVector[Double]) = {
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

    def main(args: Array[String]) {
        val f = "other/data.txt"
        val (x, y) = read_data(f)

        // Build model
        val mod = new Logistic(x, y)
        mod.run()
        println(mod.coef)
        println("# of iterations: " + mod.niter)
    }
}
