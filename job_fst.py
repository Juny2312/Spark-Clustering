#import pyspark
import pandas as pd
from pyspark.sql import SparkSession
#from pyspark import SparkContext, SparkConf
#import os
import sys
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.sql.functions import approx_count_distinct,collect_list
from pyspark.sql.functions import collect_set,sum,avg,max,countDistinct,count
from pyspark.sql.functions import first, last, kurtosis, min, mean, skewness 
from pyspark.sql.functions import stddev, stddev_samp, stddev_pop, sumDistinct
from pyspark.sql.functions import variance,var_samp,  var_pop
from pyspark.sql.functions import rank
from pyspark.sql.functions import percent_rank
from pyspark.sql.window import Window 
#from pyspark.sql.window import windowSpec

#spark = SparkSession.builder.master("local[1]").appName("kidney_disease").getOrCreate()
#data = spark.read.option("header", True).csv("data/kidney_disease.csv")  

spark = SparkSession \
        .builder \
        .getOrCreate()


windowSpec  = Window.partitionBy("LOCATION").orderBy("Value")


df_csv = spark.read\
        .option("header", True).csv("/spark/oecd/Price_level_indices.csv").cache()

#df1_csv = spark.read\
#        .option("header", True).csv("/spark/review/amazon.csv")

#df2_csv = spark.read\
#        .option("header", True).csv("/spark/review/amazon_reviews.csv")

#df3_csv = spark.read\
#        .option("header", True).csv("/spark/review/product_n_review.csv")

df1_csv = spark.read\
        .option("header", True).csv("/spark/oecd/Inflation_cpi.csv")

df2_csv = spark.read\
        .option("header", True).csv("/spark/oecd/Producer_price_indices.csv")

df3_csv = spark.read\
        .option("header", True).csv("/spark/oecd/Household_spending.csv")

df4_csv = spark.read\
        .option("header", True).csv("/spark/oecd/Household_transactions.csv")

df5_csv = spark.read\
        .option("header", True).csv("/spark/oecd/Short_term_interest.csv")

df6_csv = spark.read\
        .option("header", True).csv("/spark/oecd/Long_term_interest.csv")        


#df7_csv = spark.read\
#        .option("header", True).csv("/spark/bio32/train_taxonomy.csv")

data_pd = pd.read_csv("/home/juny/code/juny_af/etl/static_data/Price_level_indices.csv")



df_csv.printSchema()
df_csv.show()
#print(f"Row count: {df_csv.count()}")
#print(f"Column count: {len(df_csv.columns)}")
df_cols = ['LOCATION','INDICATOR','SUBJECT','TIME','Value']

df = df_csv.select(df_cols).show()
# why described as NoneType

for a in df_cols:
    df_data = df_csv.withColumn(a, df_csv[a].cast('double'))
df_data.describe(df_cols).show()
df_csv.select(collect_set("LOCATION")).show(truncate=False)
#df_csv.write.parquet("/result/parquet/df_location.parquet")

df_csv.select(collect_set("INDICATOR")).show(truncate=False)
#df_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df_lndicator.parquet")
df_csv.select(collect_set("SUBJECT")).show(truncate=False)
#df_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df_subject.parquet")
#df_csv.select(collect_set("TIME")).show(truncate=False)
#df_csv.select(collect_set("Value")).show(truncate=False)

df_csv.select(kurtosis("LOCATION")).show(truncate=False)
#df_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df_ol_location.parquet")
df_csv.select(kurtosis("INDICATOR")).show(truncate=False)
#df_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df_ol_indicator.parquet")
df_csv.select(kurtosis("SUBJECT")).show(truncate=False)
#df_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df_ol_subject.parquet")
df_csv.select(kurtosis("TIME")).show(truncate=False)
#df_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df_ol_time.parquet")
df_csv.select(kurtosis("Value")).show(truncate=False)
#df_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df_ol_value.parquet")

df_csv.select(max("TIME")).show(truncate=False)
#df_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df_max_time.parquet")
df_csv.select(min("TIME")).show(truncate=False)
#df_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df_min_time.parquet")

df_csv.select(min("Value")).show(truncate=False)
#df_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df_max_value.parquet")
df_csv.select(min("Value")).show(truncate=False)
#df_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df_min_value.parquet")

#df.select(min("salary")).show(truncate=False)
#df.select(min("salary")).show(truncate=False)
df_csv.select(stddev("Value"), stddev_samp("Value")).show(truncate=False)
#df_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df_value_std.parquet")
df_csv.select(variance("Value"),var_samp("Value")).show(truncate=False)
#df_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df_value_var.parquet")

df_csv.withColumn("price level rank",rank().over(windowSpec)) \
    .show()

#df_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df_rank.parquet")
#from pyspark.sql.window import Window
#from pyspark.sql.functions import percent_rank

df_csv.withColumn("price level percent_rank",percent_rank().over(windowSpec)) \
    .show()
#df_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df_percent_rank.parquet")







df1_csv.printSchema()
df1_csv.show()
#print(f"Row count: {df1_csv.count()}")
#print(f"Column count: {len(df1_csv.columns)}")
df1_cols = ['LOCATION','INDICATOR','SUBJECT','TIME','Value']
for b in df1_cols:
    df1_data = df1_csv.withColumn(b, df1_csv[b].cast('double'))
df1_data.describe(df1_cols).show()
df1_csv.select(collect_set("LOCATION")).show(truncate=False)
#df1.write.parquet("/home/juny/code/juny_af/etl/parquet/df1_location.parquet")
df1_csv.select(collect_set("INDICATOR")).show(truncate=False)
#df1_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df1_lndicator.parquet")
df1_csv.select(collect_set("SUBJECT")).show(truncate=False)
#df_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df1_subject.parquet")
#df1_csv.select(collect_set("TIME")).show(truncate=False)
#df1_csv.select(collect_set("Value")).show(truncate=False)

df1_csv.select(kurtosis("LOCATION")).show(truncate=False)
#df1_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df1_ol_location.parquet")
df1_csv.select(kurtosis("INDICATOR")).show(truncate=False)
#df1_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df1_ol_indicator.parquet")
df1_csv.select(kurtosis("SUBJECT")).show(truncate=False)
#df1_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df1_ol_subject.parquet")
df1_csv.select(kurtosis("TIME")).show(truncate=False)
#df1_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df1_ol_time.parquet")
df1_csv.select(kurtosis("Value")).show(truncate=False)
#df1_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df1_ol_value.parquet")

df1_csv.select(max("TIME")).show(truncate=False)
#df1_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df1_max_time.parquet")
df1_csv.select(min("TIME")).show(truncate=False)
#df1_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df1_min_time.parquet")

df1_csv.select(min("Value")).show(truncate=False)
#df1_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df1_max_value.parquet")
df1_csv.select(min("Value")).show(truncate=False)
#df1_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df1_min_value.parquet")
df1_csv.select(stddev("Value"), stddev_samp("Value")).show(truncate=False)
#df1_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df1_rank.parquet")
df1_csv.select(variance("Value"),var_samp("Value")).show(truncate=False)
#df1_csv.write.parquet("/home/juny/code/juny_af/etl/parquet/df1_percent_rank.parquet")
df1_csv.withColumn("cpi rank",rank().over(windowSpec)) \
    .show()

df1_csv.withColumn("cpi percent_rank",percent_rank().over(windowSpec)) \
    .show()
    






df2_csv.printSchema()
df2_csv.show()
#print(f"Row count: {df2_csv.count()}")
#print(f"Column count: {len(df2_csv.columns)}")
df2_cols = ['LOCATION','INDICATOR','SUBJECT','TIME','Value']
for c in df2_cols:
    df2_data = df2_csv.withColumn(c, df2_csv[c].cast('double'))
df2_data.describe(df2_cols).show()
df2_csv.select(collect_set("LOCATION")).show(truncate=False)
df2_csv.select(collect_set("INDICATOR")).show(truncate=False)
df2_csv.select(collect_set("SUBJECT")).show(truncate=False)
#df2_csv.select(collect_set("TIME")).show(truncate=False)
#df2_csv.select(collect_set("Value")).show(truncate=False)

df2_csv.select(kurtosis("LOCATION")).show(truncate=False)
df2_csv.select(kurtosis("INDICATOR")).show(truncate=False)
df2_csv.select(kurtosis("SUBJECT")).show(truncate=False)
df2_csv.select(kurtosis("TIME")).show(truncate=False)
df2_csv.select(kurtosis("Value")).show(truncate=False)

df2_csv.select(max("TIME")).show(truncate=False)
df2_csv.select(min("TIME")).show(truncate=False)

df2_csv.select(min("Value")).show(truncate=False)
df2_csv.select(min("Value")).show(truncate=False)

df2_csv.select(stddev("Value"), stddev_samp("Value")).show(truncate=False)

df2_csv.select(variance("Value"),var_samp("Value")).show(truncate=False)

df2_csv.withColumn("ppi rank",rank().over(windowSpec)) \
    .show()

df2_csv.withColumn("ppi percent_rank",percent_rank().over(windowSpec)) \
    .show()






df3_csv.printSchema()
df3_csv.show()
#print(f"Row count: {df3_csv.count()}")
#print(f"Column count: {len(df3_csv.columns)}")
df3_cols = ['LOCATION','INDICATOR','SUBJECT','TIME','Value']
for d in df3_cols:
    df3_data = df3_csv.withColumn(d, df3_csv[d].cast('double'))
df3_data.describe(df3_cols).show()
df3_csv.select(collect_set("LOCATION")).show(truncate=False)
df3_csv.select(collect_set("INDICATOR")).show(truncate=False)
df3_csv.select(collect_set("SUBJECT")).show(truncate=False)
#df3_csv.select(collect_set("TIME")).show(truncate=False)
#df3_csv.select(collect_set("Value")).show(truncate=False)

df3_csv.select(kurtosis("LOCATION")).show(truncate=False)
df3_csv.select(kurtosis("INDICATOR")).show(truncate=False)
df3_csv.select(kurtosis("SUBJECT")).show(truncate=False)
df3_csv.select(kurtosis("TIME")).show(truncate=False)
df3_csv.select(kurtosis("Value")).show(truncate=False)

df3_csv.select(max("TIME")).show(truncate=False)
df3_csv.select(min("TIME")).show(truncate=False)

df3_csv.select(min("Value")).show(truncate=False)
df3_csv.select(min("Value")).show(truncate=False)

df3_csv.select(stddev("Value"), stddev_samp("Value")).show(truncate=False)

df3_csv.select(variance("Value"),var_samp("Value")).show(truncate=False)

df3_csv.withColumn("Houshold spending rank",rank().over(windowSpec)) \
    .show()

df3_csv.withColumn("spending percent_rank",percent_rank().over(windowSpec)) \
    .show()







df4_csv.printSchema()
df4_csv.show()
#print(f"Row count: {df4_csv.count()}")
#print(f"Column count: {len(df4_csv.columns)}")
df4_cols = ['LOCATION','INDICATOR','SUBJECT','TIME','Value']
for e in df4_cols:
    df4_data = df4_csv.withColumn(e, df4_csv[e].cast('double'))
df4_data.describe(df4_cols).show()
df4_csv.select(collect_set("LOCATION")).show(truncate=False)
df4_csv.select(collect_set("INDICATOR")).show(truncate=False)
df4_csv.select(collect_set("SUBJECT")).show(truncate=False)
#df4_csv.select(collect_set("TIME")).show(truncate=False)
#df4_csv.select(collect_set("Value")).show(truncate=False)

df4_csv.select(kurtosis("LOCATION")).show(truncate=False)
df4_csv.select(kurtosis("INDICATOR")).show(truncate=False)
df4_csv.select(kurtosis("SUBJECT")).show(truncate=False)
df4_csv.select(kurtosis("TIME")).show(truncate=False)
df4_csv.select(kurtosis("Value")).show(truncate=False)

df4_csv.select(max("TIME")).show(truncate=False)
df4_csv.select(min("TIME")).show(truncate=False)

df4_csv.select(min("Value")).show(truncate=False)
df4_csv.select(min("Value")).show(truncate=False)

df4_csv.select(stddev("Value"), stddev_samp("Value")).show(truncate=False)

df4_csv.select(variance("Value"),var_samp("Value")).show(truncate=False)
df4_csv.withColumn("Household Transaction rank",rank().over(windowSpec)) \
    .show()

df4_csv.withColumn("transaction level percent_rank",percent_rank().over(windowSpec)) \
    .show()



df5_csv.printSchema()
df5_csv.show()
#print(f"Row count: {df5_csv.count()}")
#print(f"Column count: {len(df5_csv.columns)}")
df5_cols = ['LOCATION','INDICATOR','SUBJECT','TIME','Value']
for f in df5_cols:
    df5_data = df5_csv.withColumn(f, df5_csv[f].cast('double'))
df5_data.describe(df5_cols).show()
df5_csv.select(collect_set("LOCATION")).show(truncate=False)
df5_csv.select(collect_set("INDICATOR")).show(truncate=False)
df5_csv.select(collect_set("SUBJECT")).show(truncate=False)
#df5_csv.select(collect_set("TIME")).show(truncate=False)
#df5_csv.select(collect_set("Value")).show(truncate=False)

df5_csv.select(kurtosis("LOCATION")).show(truncate=False)
df5_csv.select(kurtosis("INDICATOR")).show(truncate=False)
df5_csv.select(kurtosis("SUBJECT")).show(truncate=False)
df5_csv.select(kurtosis("TIME")).show(truncate=False)
df5_csv.select(kurtosis("Value")).show(truncate=False)

df5_csv.select(max("TIME")).show(truncate=False)
df5_csv.select(min("TIME")).show(truncate=False)

df5_csv.select(min("Value")).show(truncate=False)
df5_csv.select(min("Value")).show(truncate=False)

df5_csv.select(stddev("Value"), stddev_samp("Value")).show(truncate=False)

df5_csv.select(variance("Value"),var_samp("Value")).show(truncate=False)

df5_csv.withColumn("short term interests rank",rank().over(windowSpec)) \
    .show()


df5_csv.withColumn("s-t interests  percent_rank",percent_rank().over(windowSpec)) \
    .show()




df6_csv.printSchema()
df6_csv.show()
#print(f"Row count: {df6_csv.count()}")
#print(f"Column count: {len(df6_csv.columns)}")
df6_cols = ['LOCATION','INDICATOR','SUBJECT','TIME','Value']
for g in df6_cols:
    df6_data = df6_csv.withColumn(g, df6_csv[g].cast('double'))
df6_data.describe(df6_cols).show()
df6_csv.select(collect_set("LOCATION")).show(truncate=False)
df6_csv.select(collect_set("INDICATOR")).show(truncate=False)
df6_csv.select(collect_set("SUBJECT")).show(truncate=False)
#df6_csv.select(collect_set("TIME")).show(truncate=False)
#df6_csv.select(collect_set("Value")).show(truncate=False)
#df7_csv.printSchema()
#df7_csv.show()
df6_csv.select(kurtosis("LOCATION")).show(truncate=False)
df6_csv.select(kurtosis("INDICATOR")).show(truncate=False)
df6_csv.select(kurtosis("SUBJECT")).show(truncate=False)
df6_csv.select(kurtosis("TIME")).show(truncate=False)
df6_csv.select(kurtosis("Value")).show(truncate=False)

df6_csv.select(max("TIME")).show(truncate=False)
df6_csv.select(min("TIME")).show(truncate=False)

df6_csv.select(min("Value")).show(truncate=False)
df6_csv.select(min("Value")).show(truncate=False)

df6_csv.select(stddev("Value"), stddev_samp("Value")).show(truncate=False)

df6_csv.select(variance("Value"),var_samp("Value")).show(truncate=False)

df6_csv.withColumn("long term interests rank",rank().over(windowSpec)) \
    .show()

df6_csv.withColumn("l-t interests  percent_rank",percent_rank().over(windowSpec)) \
    .show()


