scalac 01-clean-data.scala
scalac 02-standarsize-data.scala

# ~26.414s
time scala -cp $CLASSPATH:. CleanData train.csv trainLabels.csv out.txt -1
# ~106.025s
time scala -cp $CLASSPATH:. StdData out.txt clean.txt

rm *.class
