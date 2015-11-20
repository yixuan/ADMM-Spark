# ADMM-Spark

This repository is an experimental implementation of the
Alternating Direction Method of Multipliers (ADMM) algorithm on Apache Spark.
Currently the implemented model is the L1-regularized logistic regression.

## Installation

ADMM-Spark uses sbt (http://www.scala-sbt.org/index.html) to build the project
and manage the dependencies. However, the Spark library is handled
in a different way since sbt will download a lot of libraries if we specify
Spark as a dependency in the configuration file.

Instead, one can create a soft link in the `lib` directory that points to the
installed Spark `jar` file, and then run the sbt commands.

```bash
cd lib
ln -s <SPARK_HOME/lib/spark-assembly-1.5.x-hadoop2.y.z.jar> spark.jar
cd ..
sbt
```

In sbt console, type `make` to build native C++ code that helps to speed up the
computation, and then enter `package` to compile the Scala source code and
create the `jar` file.

```bash
> make
Make dynamic library from C++ code
...
> package
[info] Packaging <PROJECT_DIR>/target/scala-2.11/admm_2.11-1.0.jar ...
[info] Done packaging.
```
