package statr.stat598bd

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._

object Demo {
    private def pi(x: DenseMatrix[Double], b: DenseVector[Double]): DenseVector[Double] = {
        return 1.0 / (exp(- x * b) + 1.0)
    }

    def main(args: Array[String]) {
        // Simulate data
        val n = 100
        val p = 3
        val dist_norm = new Gaussian(0, 1)
        val x = DenseMatrix.fill(n, p) { dist_norm.draw() }
        val b = DenseVector.fill(p) { dist_norm.draw() }
        val px = pi(x, b)
        val y = px.map(x => if(new Bernoulli(x).draw()) 1.0 else 0.0)

        // Build model
        val mod = new Logistic(x, y)
        mod.run()
        println(mod.coef)
    }
}
