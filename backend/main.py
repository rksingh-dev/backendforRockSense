import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import logging
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vite's default port
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

@app.post("/api/cluster")
async def cluster_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        clusterer = ImageClusterer(n_clusters=4)
        clusterer.process_image(contents)

        # Convert the clustered image to base64
        buffered = io.BytesIO()
        clustered_image = Image.fromarray((clusterer.cluster_centers[clusterer.result] * 255).astype(np.uint8).reshape(clusterer.height, clusterer.width, clusterer.bands))
        clustered_image.save(buffered, format="PNG")
        clustered_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {
            "clusteredImage": f"data:image/png;base64,{clustered_image_base64}"
        }
    except Exception as e:
        logger.error(f"Error in cluster_image endpoint: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
