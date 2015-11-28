import scala.io.Source
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
import statr.stat598bd.PLogisticLasso

def read_y(txt: Iterator[String]): DenseVector[Double] = {
    val arr = txt.map(x => x.split(" ")(0).toDouble).toArray
    return DenseVector(arr)
}

def read_x(txt: Iterator[String]): DenseMatrix[Double] = {
    val arr = txt.map(x => x.split(" ").drop(1)).toArray
    val n = arr.length
    val p = arr(0).length
    val dat = arr.flatMap(x => x).map(x => x.toDouble)
    val mat = new DenseMatrix(p, n, dat)
    return mat.t.copy
}

val f = "file://<project path>/other/data.txt"
val txt = sc.textFile(f, 2)

sc.addJar("target/scala-2.11/admm-assembly-1.0.jar")

val daty = txt.mapPartitions(x => Array[DenseVector[Double]](read_y(x)).iterator)
val datx = txt.mapPartitions(x => Array[DenseMatrix[Double]](read_x(x)).iterator)

daty.cache()
datx.cache()

val model = new PLogisticLasso(datx, daty, sc)
model.set_lambda(2.0)
model.set_opts(500, 1e-3, 1e-3, logs = true)
model.run()
