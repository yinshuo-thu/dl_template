
df = pd.read_csv('generation_series.csv')
print(f" Original data shape: {df.shape} (number of rows, number of columns)")

df['DateTime'] = pd.to_datetime(df['reference_time_start'])
df = df.sort_values('DateTime').reset_index(drop=True)

# Basic Data Information

print("\n" + "=" * 60)
print("Data Basic Information")
print("=" * 60)
print(f"Data Shape: {df.shape[0]:,} rows × {df.shape[1]} Columns")
print(f"Time Range: {df['DateTime'].min()} to {df['DateTime'].max()}")
print(f"Total Records: {len(df):,}")
print(f"Time Span: {(df['DateTime'].max() - df['DateTime'].min()).days} days")
print("=" * 60)
