import hdbscan
import umap.umap_ as umap
import torch
from datetime import datetime
from magnet.utils.data_classes import Status
from magnet.base import Magnet
import plotly.express as px

class Clustering:
    def __init__(self, magnet: Magnet):
        self.magnet = magnet

    async def perform_umap(self, embeddings, n_components=3):
        self.magnet.status_callback(Status(datetime.now(), "info", 'Performing UMAP'))
        umap_reducer = umap.UMAP(n_components=n_components)
        umap_embeddings = umap_reducer.fit_transform(embeddings)
        self.magnet.status_callback(Status(datetime.now(), "success", "UMAP reduction completed"))
        return umap_embeddings, umap_reducer

    async def cluster_embeddings(self, umap_embeddings, min_cluster_size=10):
        self.magnet.status_callback(Status(datetime.now(), "info", 'Clustering embeddings using HDBSCAN'))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(umap_embeddings)
        self.magnet.status_callback(Status(datetime.now(), "success", "Clustering completed"))
        return cluster_labels, clusterer

    async def plot_clusters(self, umap_embeddings, cluster_labels):
        self.magnet.status_callback(Status(datetime.now(), "info", 'Plotting clusters in 3D'))
        fig = px.scatter_3d(
            x=umap_embeddings[:, 0],
            y=umap_embeddings[:, 1],
            z=umap_embeddings[:, 2],
            color=cluster_labels,
            title='HDBSCAN Clustering of Reduced Embeddings (UMAP)'
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(scene=dict(
            xaxis_title='UMAP Component 1',
            yaxis_title='UMAP Component 2',
            zaxis_title='UMAP Component 3'
        ))
        fig.show()
        self.magnet.status_callback(Status(datetime.now(), "success", "Cluster plot completed"))

    async def save_models(self, umap_reducer, clusterer, umap_export_path, clusterer_export_path):
        self.magnet.status_callback(Status(datetime.now(), "info", 'Saving UMAP and Clusterer models'))
        torch.save(umap_reducer, umap_export_path)
        torch.save(clusterer, clusterer_export_path)
        
        # Store paths in Key-Value store
        kv_store = await self.magnet.js.key_value(self.magnet.config.kv_name)
        await kv_store.put("umap_model_path", umap_export_path)
        await kv_store.put("clusterer_model_path", clusterer_export_path)
        
        self.magnet.status_callback(Status(datetime.now(), "success", "Models saved and paths logged"))
