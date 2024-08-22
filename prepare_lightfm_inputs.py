import pandas as pd
import numpy as np
from lightfm.data import Dataset
import pickle


cat_features = ["type", "colour"]
price_feature = "non_member_price"
rating_column = "liked"


def load_data(filename: str):
    df = pd.read_csv(filename)
    return df


def prepare_lightfm_df(df):
    lightfm_df = df[["product_href", "user_href", rating_column] + [price_feature] + cat_features].dropna().reset_index(drop=True).copy()

    # Create a unique list of users and products
    unique_users = lightfm_df['user_href'].unique()
    unique_products = lightfm_df['product_href'].unique()

    # Create mappings for ids
    user_id_to_href = {idx: user for idx, user in enumerate(unique_users, start=0)}
    user_href_to_id = {user: idx for idx, user in enumerate(unique_users, start=0)}
    product_id_to_href = {idx: product for idx, product in enumerate(unique_products, start=0)}
    product_href_to_id = {product: idx for idx, product in enumerate(unique_products, start=0)}

    # Map the user_href and product_href to numeric IDs
    lightfm_df['user_id'] = lightfm_df['user_href'].map(user_href_to_id)
    lightfm_df['product_id'] = lightfm_df['product_href'].map(product_href_to_id)

    # Drop the original columns if no longer needed
    lightfm_df.drop(columns=['user_href', 'product_href'], inplace=True)

    # re-order column names
    cols = ['product_id', 'user_id'] + [col for col in lightfm_df if col not in ['user_id', 'product_id']]
    lightfm_df = lightfm_df[cols]

    return lightfm_df, user_id_to_href, user_href_to_id, product_id_to_href, product_href_to_id


def separate_lightfm_df(lightfm_df):
    user_ratings_df = lightfm_df[["product_id", "user_id", rating_column]].copy()
    product_features_df = lightfm_df[["product_id"] + cat_features+[price_feature]].drop_duplicates().reset_index(drop=True).copy()
    return user_ratings_df, product_features_df


def generate_price_bins(product_features_df, price_feature):
    numeric_features_df = pd.DataFrame()
    numeric_features_df = product_features_df[["product_id", price_feature]].copy()
    # Define price bins and labels
    bins = [0, 5, 10, 15, 20, 30, 50, 100, float('inf')]
    labels = ['0-5', '5-10', '10-15', '15-20', '20-30', '30-50', '50-100', '100+']
    # Cut prices into bins and create one-hot encoded columns
    numeric_features_df['price_bin'] = pd.cut(numeric_features_df['non_member_price'].copy(), bins=bins, labels=labels, right=False)
    price_bins = pd.get_dummies(numeric_features_df['price_bin'], prefix='price_bin')
    # Concatenate the one-hot encoded columns back to the original DataFrame
    numeric_features_df = pd.concat([numeric_features_df, price_bins], axis=1)
    numeric_features_df = numeric_features_df.drop(columns=["non_member_price", "price_bin"]).astype(int)
    return numeric_features_df


def encode_types(product_features_df):
    type_features_df = pd.DataFrame()
    type_features_df = product_features_df[["product_id", "type"]].copy()
    type_features_df['type'] = type_features_df['type'].apply(lambda x: x.split(', '))
    type_features_df = type_features_df.explode('type')
    type_features_df = pd.get_dummies(type_features_df, columns=['type'])
    type_features_df = type_features_df.astype(int)
    type_features_df = type_features_df.groupby(['product_id']).sum().reset_index().astype(int)
    return type_features_df


def select_non_type_cat_features(product_features_df, non_type_cat_features):
    cat_features_df = pd.DataFrame()
    cat_features_df = product_features_df[["product_id"] + non_type_cat_features].copy()
    cat_features_df = pd.get_dummies(cat_features_df, columns=non_type_cat_features).astype(int)
    return cat_features_df


def merge_encoded_product_features(numeric_features_df, type_features_df, cat_features_df):
    # Initialize list of DataFrames to merge
    dfs_to_merge = []

    # Append DataFrames to the list if they are not empty
    if not numeric_features_df.empty:
        dfs_to_merge.append(numeric_features_df)
    if not type_features_df.empty:
        dfs_to_merge.append(type_features_df)
    if not cat_features_df.empty:
        dfs_to_merge.append(cat_features_df)

    # Perform the merge
    if dfs_to_merge:
        encoded_product_features_df = dfs_to_merge[0]
        for df in dfs_to_merge[1:]:
            encoded_product_features_df = pd.merge(encoded_product_features_df, df, on='product_id', how='inner')

    return encoded_product_features_df


def encode_product_features(product_features_df):
    if len(price_feature) > 0:
        numeric_features_df = generate_price_bins(product_features_df, price_feature)
    
    if "type" in cat_features:
        type_features_df = encode_types(product_features_df)
    
    non_type_cat_features = [col for col in cat_features if col != 'type']
    if len(non_type_cat_features) > 0:
        cat_features_df = select_non_type_cat_features(product_features_df, non_type_cat_features)

    encoded_product_features_df = merge_encoded_product_features(numeric_features_df, type_features_df, cat_features_df)

    return encoded_product_features_df


def replace_binary_encodings(encoded_product_features_df):
    # Replaces the 1/0 values with the column names. This is needed for processing the features for LightFM
    product_features_tags_df = encoded_product_features_df.copy()
    for x in product_features_tags_df.drop(columns=["product_id"]).columns:
        product_features_tags_df[x] = np.where(product_features_tags_df[x]==1, x,"")
    return product_features_tags_df


## Get unique list of item metadata
def get_item_metadata(product_features_tags_df):
    item_metadata = product_features_tags_df.drop(columns=["product_id"]).columns
    return item_metadata


### prepare dataset object for lightfm model
def fit_dataset(lightfm_df, item_metadata):
    dataset = Dataset()

    dataset.fit(
        users=lightfm_df['user_id'].unique(),
        items=lightfm_df['product_id'].unique(),
        item_features = item_metadata
    )

    return dataset


# Create lookup of items to their item features
def create_item_lookup(product_features_tags_df):
    item_to_metadata_lookup = [(x[0], list(filter(None, x[1:]))) for x in product_features_tags_df.values]
    return item_to_metadata_lookup


# Create weights for categorical product features
def create_weighted_item_lookup(item_to_metadata_lookup, priority_feature="colour"):
    weighted_item_to_metadata_lookup = []

    # add weights
    for product_feature_map in item_to_metadata_lookup:
        feature_list = product_feature_map[1]
        feature_dict = {}
        for feature in feature_list:
            if priority_feature in feature:
                feature_dict[feature] = 2
            else:
                feature_dict[feature] = 1
        weighted_product_feature_map = (product_feature_map[0], feature_dict)
        weighted_item_to_metadata_lookup.append(weighted_product_feature_map)
    return weighted_item_to_metadata_lookup

# Create item feature + tag list with LightFM function
def create_item_metadata_list(dataset, weighted_item_to_metadata_lookup):
    item_metadata_list = dataset.build_item_features(weighted_item_to_metadata_lookup, normalize=True)
    return item_metadata_list


# Increase the target weights to emphasize explicit feedback
def scale_liked(rating):
    if rating == 0:
        return 0  # Small positive weight for 'not liked'
    elif rating == 1:
        return 5  # Higher weight for 'liked'
    elif rating == 2:
        return 10  # Highest weight for 'really liked'


# Build interactions (user — item) and their respective weights
def build_interactions(user_ratings_df, dataset):
    # Apply the function to create a new column with scaled ratings
    user_ratings_df['weight'] = user_ratings_df['liked'].apply(scale_liked)

    # Build the interactions matrix with ratings as weights
    interactions, weights = dataset.build_interactions(user_ratings_df[['user_id', 'product_id', "weight"]].values)

    return interactions, weights

def save_pickle(filename, object):

    # Open the file in binary write mode ('wb')
    with open(filename, 'wb') as file:
        # Use pickle.dump() to write the data to the file
        pickle.dump(object, file)

    print(f"saved {filename}")

def prepare_inputs():
    df = load_data("data/training_df.csv")
    lightfm_df, user_id_to_href, user_href_to_id, product_id_to_href, product_href_to_id = prepare_lightfm_df(df)
    user_ratings_df, product_features_df = separate_lightfm_df(lightfm_df)
    encoded_product_features_df = encode_product_features(product_features_df)
    product_features_tags_df = replace_binary_encodings(encoded_product_features_df)
    item_metadata = get_item_metadata(product_features_tags_df)
    dataset = fit_dataset(lightfm_df, item_metadata)
    item_to_metadata_lookup = create_item_lookup(product_features_tags_df)
    weighted_item_to_metadata_lookup = create_weighted_item_lookup(item_to_metadata_lookup)
    item_metadata_list = create_item_metadata_list(dataset, weighted_item_to_metadata_lookup)
    interactions, weights = build_interactions(user_ratings_df, dataset)

    # Output interactions, weights, item_metadata_list, 
    save_pickle("data/lightfm_inputs/interactions.pkl", interactions)
    save_pickle("data/lightfm_inputs/weights.pkl", weights)
    save_pickle("data/lightfm_inputs/item_metadata_list.pkl", item_metadata_list)

    # Output user_id_to_href, user_href_to_id, product_id_to_href, product_href_to_id
    save_pickle("data/lightfm_mappings/user_id_to_href.pkl", user_id_to_href)
    save_pickle("data/lightfm_mappings/user_href_to_id.pkl", user_href_to_id)
    save_pickle("data/lightfm_mappings/product_id_to_href.pkl", product_id_to_href)
    save_pickle("data/lightfm_mappings/product_href_to_id.pkl", product_href_to_id)

prepare_inputs()