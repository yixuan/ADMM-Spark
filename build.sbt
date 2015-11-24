lazy val make = taskKey[Unit]("Make dynamic library from C++ code")
lazy val jni = taskKey[Unit]("Generate JNI headers")

lazy val root = (project in file(".")).
  settings(
    name := "admm",
    version := "1.0",
    scalaVersion := "2.11.7",
    libraryDependencies ++= Seq(
      // Testing facility
      "org.scalatest" % "scalatest_2.11" % "2.2.4" % "test",
      // Spark core components
      "org.apache.spark" %% "spark-core" % "1.5.2" % "provided",
      // For matrix decomposition using pure Java
      "org.ejml" % "core" % "0.28",
      "org.ejml" % "dense64" % "0.28",
      // JNI loader
      "com.github.fommil" % "jniloader" % "1.1",
      // Matrix operations using Breeze
      "org.scalanlp" %% "breeze" % "0.11.2" % "provided", // included by Spark
      // native libraries are not included by default. add this if you want them (as of 0.7)
      // native libraries greatly improve performance, but increase jar sizes.
      // It also packages various blas implementations, which have licenses that may or may not
      // be compatible with the Apache License. No GPL code, as best I know.
      "org.scalanlp" %% "breeze-natives" % "0.11.2" % "provided"
      // the visualization library is distributed separately as well.
      // It depends on LGPL code.
      // "org.scalanlp" %% "breeze-viz" % "0.11.2"
    ),
    resolvers ++= Seq(
      // other resolvers here
      // if you want to use snapshot builds (currently 0.12-SNAPSHOT), use this.
      "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
      "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
    )
  ).
  settings(
    make := {
      println("Make dynamic library from C++ code")
      "sh make.sh" !
    },
    jni := {
      "sh jni.sh" !
    }
  )

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)

excludeFilter in unmanagedSources := HiddenFileFilter || "demo.scala"
