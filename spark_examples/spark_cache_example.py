import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType

# Create a spark session and dataframe
spark = SparkSession.builder.appName('my_spark_session').getOrCreate()
spark.sparkContext.setLogLevel("INFO")  # INFO is default
pandas_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
TABLE_SCHEMA = StructType([
    StructField('a', IntegerType(), False),
    StructField('b', IntegerType(), False)])
df = spark.createDataFrame(pandas_df, schema=TABLE_SCHEMA)
print(id(df))
print(df)
print('\n')

# Cache the dataframe and store a reference
my_cache = df
df.cache()

# Make a change (Changes in pyspark are NOT made inplace. Hence, we create a new object.)
df = df.withColumn('c', F.lit(6))
print(id(df))
print(df)
print('\n')

# Take the cached object, and print it to verify it is unchanged
print(id(my_cache))
print(my_cache)
print('\n')
