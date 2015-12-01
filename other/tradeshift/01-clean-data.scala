import scala.io.Source
import java.io._
import scala.util.control._


// scala -cp . CleanData train.csv trainLabels.csv out.txt <nlines>
// nlines <= 0 means reading all
object CleanData {
    def remove_text(line: String): Array[String] = {
        return line.split(',').filter(_.length < 30)
    }

    def encode(ch: String): String = ch match {
        case "YES" => "1"
        case "NO"  => "0"
        case vals  => vals
    }

    def main(args: Array[String]) {
        if (args.length != 4) {
            return
        }

        val infile  = args(0)
        val label   = args(1)
        val outfile = args(2)
        val nlines  = args(3).toInt

        val data_src  = Source.fromFile(infile)
        val label_src = Source.fromFile(label)
        val dest = new PrintWriter(new File(outfile))

        val data_iter  = data_src.getLines()
        val label_iter = label_src.getLines()

        // Skip the first line
        data_iter.next
        label_iter.next

        var i = 0
        val loop = new Breaks
        loop.breakable {
            for (line <- data_iter) {
                // X data
                val x = remove_text(line)
                // Y data
                val y = label_iter.next
                // Do not write lines that contain missing values
                if (!(x.contains(""))) {
                    // Write y33 in the first column
                    dest.write(y.last + " ")
                    // Write x data (excluding id) separated by white space
                    dest.println(x.drop(1).map(encode).mkString(" "))
                }

                i += 1
                if (nlines > 0 && i >= nlines)
                    loop.break()
            }
        }

        data_src.close()
        label_src.close()
        dest.close()
    }
}
