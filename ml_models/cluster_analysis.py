from sklearn.cluster import KMeans
import pandas as pd

def detect_congestion_clusters(df, n_clusters=4):
    features = df[['temp', 'clouds_all', 'traffic_volume']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(features)
    return df[['date_time', 'traffic_volume', 'cluster']]

def detect_congestion_clusters_in_chunks(df, chunk_size=500):
    cluster_results = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        clustered_df = detect_congestion_clusters(chunk)
        cluster_results.append(clustered_df)
    return cluster_results
