import { useDropzone } from 'react-dropzone';
import { Upload, ImageIcon, BarChart3, PieChart, Activity, Loader2 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart as RePieChart, Pie, Cell } from 'recharts';
import axios from 'axios';
import { useState, useCallback } from 'react';

interface ClusterData {
  id: number;
  pixelCount: number;
  meanColor: [number, number, number];
}

function App() {
  const [image, setImage] = useState<string | null>(null);
  const [clusteredImage, setClusteredImage] = useState<string | null>(null);
  const [clusters, setClusters] = useState<ClusterData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setLoading(true); 
      setError(null);

      // Show original image
      const reader = new FileReader();
      reader.onload = () => {
        setImage(reader.result as string);
      };
      reader.readAsDataURL(file);

      // Send to backend
      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await axios.post('http://localhost:8000/api/cluster', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });

        setClusteredImage(response.data.clusteredImage);
        setClusters(response.data.clusters);
      } catch (err) {
        setError('Failed to process image. Please try again.');
        console.error('Error:', err);
      } finally {
        setLoading(false);
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    multiple: false
  });

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8 flex items-center">
          <ImageIcon className="h-8 w-8 text-indigo-600" />
          <h1 className="ml-3 text-2xl font-bold text-gray-900">Mineral Clustering Analysis</h1>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {/* Upload Section */}
        <div {...getRootProps()} className="mb-8">
          <input {...getInputProps()} />
          <div className={`border-2 border-dashed rounded-lg p-12 text-center ${
            isDragActive ? 'border-indigo-500 bg-indigo-50' : 'border-gray-300'
          }`}>
            {loading ? (
              <div className="flex flex-col items-center">
                <Loader2 className="h-12 w-12 text-indigo-500 animate-spin" />
                <p className="mt-2 text-sm text-gray-600">Processing image...</p>
              </div>
            ) : (
              <>
                <Upload className="mx-auto h-12 w-12 text-gray-400" />
                <p className="mt-2 text-sm text-gray-600">
                  Drag and drop your mineral image here, or click to select a file
                </p>
              </>
            )}
          </div>
        </div>

        {error && (
          <div className="mb-8 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded relative">
            {error}
          </div>
        )}

        {/* Results Grid */}
        {image && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Original Image */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-lg font-semibold mb-4">Original Image</h2>
              <img src={image} alt="Original" className="w-full rounded" />
            </div>

            {/* Clustered Image */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-lg font-semibold mb-4">Clustered Result</h2>
              {clusteredImage ? (
                <img src={clusteredImage} alt="Clustered" className="w-full rounded" />
              ) : (
                <div className="aspect-video bg-gray-100 rounded flex items-center justify-center">
                  <p className="text-gray-500">Processing...</p>
                </div>
              )}
            </div>

            {/* Cluster Distribution */}
            {clusters.length > 0 && (
              <div className="bg-white p-6 rounded-lg shadow">
                <h2 className="text-lg font-semibold mb-4">Cluster Distribution</h2>
                <RePieChart width={400} height={300}>
                  <Pie
                    data={clusters}
                    dataKey="pixelCount"
                    nameKey="id"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label
                  >
                    {clusters.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </RePieChart>
              </div>
            )}

            {/* Color Analysis */}
            {clusters.length > 0 && (
              <div className="bg-white p-6 rounded-lg shadow">
                <h2 className="text-lg font-semibold mb-4">Color Analysis</h2>
                <BarChart width={400} height={300} data={clusters}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="id" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="meanColor[0]" name="Red" fill="#ff0000" />
                  <Bar dataKey="meanColor[1]" name="Green" fill="#00ff00" />
                  <Bar dataKey="meanColor[2]" name="Blue" fill="#0000ff" />
                </BarChart>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;