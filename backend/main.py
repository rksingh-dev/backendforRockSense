from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple
import logging
import io
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins temporarily for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageClusterer:
    """A class to perform K-means clustering on images."""

    def __init__(self, n_clusters: int = 7):
        self.n_clusters = n_clusters
        self.height = None
        self.width = None
        self.bands = None
        self.image_array = None
        self.result = None
        self.cluster_centers = None

    def process_image(self, image_data):
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            im = np.array(image)

            self.height = im.shape[0]
            self.width = im.shape[1]
            self.bands = im.shape[2]
            self.image_array = im.reshape(-1, self.bands)

            logger.info(f"Image loaded successfully. Size: {im.shape}")
            self.perform_clustering()
            self.visualize_results()
            self.plot_histograms()
            self.plot_color_distribution()
            self.plot_cluster_composition()
            self.plot_cumulative_distribution()
            self.plot_correlation_matrix()
            self.cluster_statistics()
            self.save_outputs()

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            print(f"Error processing image: {str(e)}")

    def perform_clustering(self) -> np.ndarray:
        try:
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
            kmeans.fit(self.image_array)
            self.result = kmeans.labels_.reshape(self.height, self.width)
            self.cluster_centers = kmeans.cluster_centers_
            logger.info("Clustering completed successfully")
            return self.result

        except Exception as e:
            logger.error(f"Error during clustering: {str(e)}")
            raise

    def save_outputs(self) -> None:
        """Save output images and statistics to files."""
        try:
            # Save clustered image
            clustered_image = Image.fromarray((self.cluster_centers[self.result] * 255).astype(np.uint8).reshape(self.height, self.width, self.bands))
            clustered_image.save('clustered_image.png')

            # Save statistics
            with open('cluster_statistics.txt', 'w') as f:
                for i in range(self.n_clusters):
                    cluster_mask = self.result.flatten() == i
                    cluster_pixels = self.image_array[cluster_mask]
                    if len(cluster_pixels) > 0:
                        mean_color = np.mean(cluster_pixels, axis=0)
                        std_color = np.std(cluster_pixels, axis=0)
                        median_color = np.median(cluster_pixels, axis=0)
                        f.write(f"Cluster {i}:\n")
                        f.write(f"  Mean (R,G,B): {mean_color.tolist()}\n")
                        f.write(f"  Standard Deviation (R,G,B): {std_color.tolist()}\n")
                        f.write(f"  Median (R,G,B): {median_color.tolist()}\n")
                        f.write(f"  Pixel Count: {len(cluster_pixels)}\n\n")

            logger.info("Outputs saved successfully")

        except Exception as e:
            logger.error(f"Error saving outputs: {str(e)}")
            raise

# Add your API endpoints here
