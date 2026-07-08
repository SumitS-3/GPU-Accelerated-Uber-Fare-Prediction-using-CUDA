import pyarrow.parquet as pq

table = pq.read_table(r"Data\yellow_tripdata_2024-01.parquet")
print("Schema names:", table.schema.names)
print("Row count:", table.num_rows)
