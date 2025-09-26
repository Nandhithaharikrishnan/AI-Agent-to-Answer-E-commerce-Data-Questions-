import os
import pandas as pd
from sqlalchemy import create_engine

# Set up paths
data_dir = os.path.join(os.path.dirname(__file__), 'data')
db_path = os.path.join(os.path.dirname(__file__), 'data.db')
engine = create_engine(f'sqlite:///{db_path}')

# List CSV files in the data directory
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

for csv_file in csv_files:
    table_name = os.path.splitext(csv_file)[0].replace(' ', '_').replace('-', '_').lower()
    csv_path = os.path.join(data_dir, csv_file)
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"Loaded {csv_file} into table '{table_name}'")

print("All CSV files loaded into SQLite database at data.db.")