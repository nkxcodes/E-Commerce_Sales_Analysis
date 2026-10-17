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

    # Data Checking and Data Cleaning
    missing_values = df.isnull().sum()
    missing_rows = df[df.isnull().any(axis=1)]
    duplicate_rows = df.duplicated().sum()
    duplicate_orders = df[df.duplicated(subset='order_id', keep=False)]

    print()
    print(missing_rows)

    print('Exact duplicate rows: ')
    print(duplicate_rows)

    print(df[df.duplicated()])

    clean_df = clean_df.drop_duplicates()

    print('Duplicate orders: ')
    print(duplicate_orders)

    clean_df['order_date'] = pd.to_datetime(
        clean_df['order_date'],
        errors = 'coerce'
    )

    print(clean_df['order_date'].dtype)

    invalid_dates = clean_df[
        clean_df['order_date'].isna()
    ]

    print(invalid_dates)

    print(clean_df['city'].unique())
    print(clean_df['category'].unique())
    print(clean_df['payment_method'].unique())

    text_columns = [
        'product',
        'category',
        'city',
        'payment_method',
        'customer_id'
    ]

    for column in text_columns:
        clean_df[column] = clean_df[column].str.strip()

    clean_df['city'] = clean_df['city'].str.title()
    clean_df['category'] = clean_df['category'].str.title()
    clean_df['payment_method'] = clean_df['payment_method'].str.title()

    invalid_quantity = clean_df[
        clean_df['quantity'].notna()
        & (clean_df['quantity'] <= 0)
    ]

    print(invalid_quantity)

    print(clean_df['unit_price'].describe())

    invalid_prices = clean_df[
        clean_df['unit_price'].notna()
        & (clean_df['unit_price'] <= 0)
    ]

    print(invalid_prices)

    suspicious_prices = clean_df[
        clean_df['unit_price'] > 10000
    ]

    print(suspicious_prices)

    print(f'Original dataset shape: {df.shape}')
    print(f'Working dataset shape: {clean_df.shape}')

    print('Original missing values: ')
    print(missing_values)

    print('Working dataset missing values: ')
    print(clean_df.isnull().sum())

    print('Remaining exact duplicate rows: ')
    print(clean_df.duplicated().sum())

    print('========== KEY PERFORMANCE INDICATORS (KPIs) ==========')
    total_revenue = (clean_df['quantity'] * clean_df['unit_price']).sum()
    total_orders = clean_df['order_id'].count()
    units_sold = clean_df['quantity'].sum()
    average_order_value = total_revenue / total_orders
    average_selling_price = clean_df['unit_price'].mean()
    best_selling_product = clean_df.loc[
        clean_df['quantity'].idxmax(),
        'product'
    ]
    top_revenue_category = clean_df.groupby('category')['unit_price'].sum().idxmax()

    print()
    print(f'Total Revenue: {total_revenue}')

    print()
    print(f'Total Orders: {total_orders}')

    print()
    print(f'Units Sold: {units_sold}')

    print()
    print(f'Average Order Value: {average_order_value}')

    print()
    print(f'Average Selling Price: {average_selling_price}')

    print()
    print(f'Best Selling Product: {best_selling_product}')

    print()
    print(f'Top Revenue Category: {top_revenue_category}')

if __name__ == '__main__':
    main()