import scala.io.Source
import java.io._
import breeze.linalg._
import breeze.numerics._


// scala -cp $CLASSPATH:. StdData out.txt clean.txt
object StdData {

    def main(args: Array[String]) {
        if (args.length != 2) {
            return
        }

        val infile  = args(0)
        val outfile = args(1)

        var source  = Source.fromFile(infile)
        val n = source.getLines().length.toDouble
        source.close()

        source  = Source.fromFile(infile)
        val iter = source.getLines()
        val first_line = iter.next
        // x bar
        val means = new DenseVector(first_line.split(' ').drop(1).map(x => x.toDouble))
        // x^2 bar
        val meanss = means :* means
        means :/= n
        meanss :/= n

        for (line <- iter) {
            val curr = new DenseVector(line.split(' ').drop(1).map(x => x.toDouble))
            means :+= (curr :/ n)
            meanss :+= (curr :* curr :/ n)
        }
        source.close()
        val scale = sqrt(meanss - (means :* means))

        source  = Source.fromFile(infile)
        val dest = new PrintWriter(new File(outfile))
        for (line <- source.getLines()) {
            val curr = new DenseVector(line.split(' ').drop(1).map(x => x.toDouble))
            val trans = (curr - means) :/ scale
            // y value, intercept for x, x values
            dest.println(line.take(1) + " 1 " + trans.data.mkString(" "))
        }
        source.close()
        dest.close()
    }
}
