ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.8"

lazy val root = (project in file("."))
  .settings(
    name := "spark-assignment-2"
  )

libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.2.4"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.2.4" % "provided"

