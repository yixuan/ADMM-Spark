# ADMM-Spark

This repository is an experimental implementation of the
Alternating Direction Method of Multipliers (ADMM) algorithm on Apache Spark.
Currently the implemented model is the L1-regularized logistic regression.

## Dependencies

ADMM-Spark relies on the following libraries:

- Apache Spark (http://spark.apache.org/)
- Breeze (https://github.com/scalanlp/breeze, already included in Spark)
- Efficient Java Matrix Library (http://ejml.org/)

The sbt building tool will handle these dependencies. See the next section
for details.

## Installation

ADMM-Spark uses sbt (http://www.scala-sbt.org/index.html) to build the project
and manage the dependencies. To create the `jar` file, first enter the project
directory, and then type the `sbt` command.

```bash
cd ADMM-Spark
sbt
```

In sbt console, input `make` to build native C++ code that helps to speed up the
computation, and then enter `package` to compile the Scala source code and
create the `jar` file. If one also wants the `jar` file to include the
dependent libraries so that it can be directly used in Spark applications,
the `assembly` command can be used instead.

```bash
> make
Make dynamic library from C++ code
...
> package
[info] Packaging <PROJECT_DIR>/target/scala-2.11/admm_2.11-1.0.jar ...
[info] Done packaging.
> assembly
...
[info] Done packaging.
```
