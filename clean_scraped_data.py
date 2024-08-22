import pandas as pd

def load_data(filename: str):
    df = pd.read_csv(filename)
    return df

def combine_user_reviews_dfs(vivinio_reviews_df, dm_reviews_df):
    vivinio_reviews_df = vivinio_reviews_df[["dm_href", "user_href", "username", "rating"]]
    dm_reviews_df = dm_reviews_df[["product_href", "user_href", "username", "rating"]]
    vivinio_reviews_df = vivinio_reviews_df.rename(columns={'dm_href': 'product_href'})
    user_review_df = pd.concat([vivinio_reviews_df, dm_reviews_df], ignore_index=True)
    return user_review_df

def merge_product_users_dfs(wine_features_df, user_review_df):
    wine_features_df = wine_features_df.rename(columns={'href': 'product_href'})
    training_df = user_review_df.merge(wine_features_df, on="product_href")
    return training_df

def filter_by_count(df, threshold: int = 8):
    user_counts = df["user_href"].value_counts()
    valid_users = user_counts[user_counts >= threshold].index.tolist()
    filtered_df = df[df["user_href"].isin(valid_users)].reset_index(drop=True).copy()
    return filtered_df

def clean_colour(df):
    df.loc[df["type"] == "rose", "colour"] = "rose"
    df['colour'] = df['colour'].str.lower()
    return df

def map_rating_to_like(rating):
    if rating > 4.:
        return 2
    elif rating >= 3.5:
        return 1
    else:
        return 0
    
def generate_liked_column(df):
    df['liked'] = df['rating'].apply(map_rating_to_like)
    return df

def clean_body_labels(df):
    df['body'] = df['body'].str.lower()
    df['body'] = df['body'].replace('light', 'light bodied')
    df['body'] = df['body'].replace('medium-bodied', 'medium bodied')
    df['body'] = df['body'].replace('mid-bodied', 'medium bodied')
    df['body'] = df['body'].replace('full', 'full bodied')
    df['body'] = df['body'].replace('heavy', 'full bodied')
    df['body'] = df['body'].replace('cabernet sauvignon', 'full bodied')
    df['body'] = df['body'].replace('unashamedly full-bodied shiraz', 'full bodied')
    return df

def clean_sweetness_column(df):
    mapping = {
        'dry': 'dry',
        'sweet': 'sweet',
        'semi dry': 'semi dry',
        'semi sweet': 'semi sweet',
        'medium bodied': 'null',
        'full bodied': 'null',
        'riesling': 'null',
        'red': 'null',
        'white': 'null',
        'shiraz': 'null',
        'light bodied': 'null',
        'cabernet merlot': 'null',
        'extra dry': 'null',
        'screw cap': 'null',
        'pinot noir': 'null',
        'off dry': 'null',
        'luscious': 'null',
        'dry rosé': 'null'
    }
    df['sweetness'] = df['sweetness'].str.lower()
    df['sweetness'] = df['sweetness'].map(mapping).fillna('null')
    return df

def clean_scraped_data():
    # load dfs
    wine_features_df = load_data("data/wine_features.csv")
    vivinio_reviews_df = load_data("data/vivino_user_reviews.csv")
    dm_reviews_df = load_data("data/dm_user_reviews.csv")

    # merge dfs
    user_review_df = combine_user_reviews_dfs(vivinio_reviews_df, dm_reviews_df)
    training_df = merge_product_users_dfs(wine_features_df, user_review_df)
    training_df = filter_by_count(training_df)

    #clean columns
    training_df = clean_colour(training_df)
    training_df = generate_liked_column(training_df)
    training_df = clean_body_labels(training_df)
    training_df = clean_sweetness_column(training_df)

    # save to csv
    training_df.to_csv("data/training_df.csv", index=False)

clean_scraped_data()