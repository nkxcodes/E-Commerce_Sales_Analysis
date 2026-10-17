import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('data/ecommerce_orders.csv')
    clean_df = df.copy()

    first_five_rows = df.head()
    last_five_rows = df.tail()
    column_names = df.columns
    data_types = df.dtypes

    print(first_five_rows)

    print()
    print(last_five_rows)

    print()
    print(column_names)

    print()
    print(data_types)

if __name__ == '__main__':
    main()