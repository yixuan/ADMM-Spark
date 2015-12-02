scalac 01-clean-data.scala

# ~26.414s
time scala -cp $CLASSPATH:. CleanData train.csv trainLabels.csv out.txt -1

rm *.class

hdfs dfs -put out.txt tradeshift

hdfs dfs -ls
