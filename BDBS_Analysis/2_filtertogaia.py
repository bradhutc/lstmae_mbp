import pandas as pd
import pyarrow.parquet as pq

file_path = '/N/project/catypGC/BDBS/newprocessed_data.parquet'

table = pq.read_table(file_path, columns=None, use_pandas_metadata=True)

desired_rows = 5000000

df = table.to_pandas().head(desired_rows)

# Save the DataFrame to CSV
df.to_csv('gaiaupload.csv', index=False)

print(df.head())
print(len(df))
