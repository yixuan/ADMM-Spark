// spark-shell --master yarn-client --num-executors 10

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.log4j.Logger
import org.apache.log4j.Level

import breeze.linalg._
import breeze.numerics._
import statr.stat598bd.PLogisticLasso

object TradeShift {

    def read_data(txt: Iterator[String]): (DenseVector[Double], DenseMatrix[Double]) = {
        val parsed = txt.map(x => x.split(' ').map(_.toDouble)).toArray

        val y = parsed.map(x => x(0))

        val n = parsed.length
        val p = parsed(0).length - 1
        val x = parsed.map(x => x.drop(1)).flatMap(x => x)
        val mat = new DenseMatrix(p, n, x)

        return (new DenseVector(y), mat.t.copy)
    }

    def lambdas(n: Int, minr: Double = 0.001, maxr: Double = 1.0, nlambda: Int = 10): Array[Double] = {
        val logmin = math.log(minr * n)
        val logmax = math.log(maxr * n)
        val step = (logmax - logmin) / (nlambda - 1.0)
        val lambda = new Array[Double](nlambda)
        for(i <- 0 until nlambda) {
            lambda(i) = math.exp(logmax - i * step)
        }
        return lambda
    }

    def recover_beta(beta: SparseVector[Double],
                     center: DenseVector[Double],
                     scale: DenseVector[Double]): (Double, SparseVector[Double]) = {
        var intercept = beta(0)
        val builder = new VectorBuilder[Double](beta.length - 1)
        var offset = 1
        while(offset < beta.activeSize) {
            val index = beta.indexAt(offset)
            val value = beta.valueAt(offset)
            builder.add(index - 1, value / scale(index - 1))
            intercept -= value * center(index - 1) / scale(index - 1)

            offset += 1
        }
        return (intercept, builder.toSparseVector(true, true))
    }

    def predict(intercept: Double,
                coef: SparseVector[Double],
                data: RDD[DenseMatrix[Double]]): RDD[DenseVector[Double]] = {
        data.map(x => 1.0 / (exp(-(x * coef) - intercept) + 1.0))
    }

    def loglikelihood(pred: RDD[DenseVector[Double]],
                         y: RDD[DenseVector[Double]]): Double = {
        return pred.zip(y).map(s =>
            sum((s._2 :* log(s._1)) + ((1.0 - s._2) :* log(1.0 - s._1)))
        ).reduce(_ + _)
    }

    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("Tradeshift Example")
        val sc = new SparkContext(conf)

        val f = "hdfs://path/to/data"
        val txt = sc.textFile(f, 10)

        // Seed for random split
        val seed = 123
        // Split sample
        val sets = txt.randomSplit(Array(0.99, 0.01), seed)
        val train_set = sets(0)
        val tune_set = sets(1)

        // Read data
        var t1 = System.currentTimeMillis()
        val train = train_set.mapPartitions(x => Array[(DenseVector[Double], DenseMatrix[Double])](read_data(x)).iterator)
        val tune = tune_set.mapPartitions(x => Array[(DenseVector[Double], DenseMatrix[Double])](read_data(x)).iterator)
        // Cache to memory
        train.cache()
        tune.cache()
        train.count  // To trigger the caching
        tune.count   // To trigger the caching
        var t2 = System.currentTimeMillis()
        println("Reading data: " + (t2 - t1) / 1000.0 + "s")

        // Sample size
        val n = train.map(x => x._2.rows).reduce(_ + _)

        // Calculate centering and scaling factor
        t1 = System.currentTimeMillis()
        val xbar = train.map(x => (sum(x._2(::, *)) / n.toDouble).toDenseVector).reduce(_ + _)
        val xsqbar = train.map(x => {
            val xx = x._2 :* (x._2 :/ n.toDouble)
            sum(xx(::, *)).toDenseVector
        }).reduce(_ + _)
        val scale = sqrt(xsqbar - (xbar :* xbar))

        // Standardized data
        val train_std = train.map(x => {
            val xcenter = x._2(*, ::) - xbar
            val xstd = xcenter(*, ::) :/ scale
            (x._1, DenseMatrix.horzcat(DenseVector.ones[Double](x._2.rows).toDenseMatrix.t, xstd))
        })
        train_std.cache()
        train_std.count  // To trigger the caching
        t2 = System.currentTimeMillis()
        println("Standardizing data: " + (t2 - t1) / 1000.0 + "s")

        // Setting up model
        sc.addJar("admm-assembly-1.0.jar")
        t1 = System.currentTimeMillis()
        val model = new PLogisticLasso(train_std.map(_._2), train_std.map(_._1), sc)
        model.set_opts(100, 1e-3, 1e-3, logs = true)
        t2 = System.currentTimeMillis()
        println("Setting up model: " + (t2 - t1) / 1000.0 + "s")

        // Generate grids of lambdas
        val lambda = lambdas(n, 0.001, 0.1)
        val neglll = new DenseVector[Double](lambda.length)
        val betas = new Array[(Double, SparseVector[Double])](lambda.length)

        for(i <- 0 until lambda.length) {
            println("\nlambda = " + lambda(i) + "\n")

            t1 = System.currentTimeMillis()
            model.set_lambda(lambda(i))
            model.run()
            t2 = System.currentTimeMillis()
            println("Iteration time: " + (t2 - t1) / 1000.0 + "s")

            // Get estimated coefficients
            betas(i) = recover_beta(model.coef, xbar, scale)
            // Predicted probabilities
            val pred = predict(betas(i)._1, betas(i)._2, tune.map(_._2))
            // Negtive log-likelihood on tuning set
            neglll(i) = -loglikelihood(pred, tune.map(_._1))
            println("Neg-log-likelihood = " + neglll(i))
        }

        train_std.unpersist()
        tune.unpersist()

        val ind = argmin(neglll)
        val predt = predict(betas(ind)._1, betas(ind)._2, train.map(_._2))
        val ypred = predt.flatMap(x => x.data.map(s => if(s < 0.5) 0.0 else 1.0))
        val ytrue = train.flatMap(_._1.data)
        val accuracy = ypred.zip(ytrue).map(x => if(x._1 == x._2) 1.0 else 0.0).reduce(_ + _) / n.toDouble

        println("Accuracy on training set = " + accuracy)
    }
}
