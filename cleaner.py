def clean_and_merge(sales, features, stores):
    features['Date'] = pd.to_datetime(features['Date'])
    sales['Date'] = pd.to_datetime(sales['Date'])

    df = sales.merge(features, on=['Store', 'Date'], how='left')
    df = df.merge(stores, on='Store', how='left')

    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in markdown_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)

    df['CPI'] = df['CPI'].fillna(method='ffill')
    df['Unemployment'] = df['Unemployment'].fillna(method='ffill')
    
    return df
