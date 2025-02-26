import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Loader2 } from 'lucide-react';
import axios from 'axios';

function App() {
  const [clusteredImage, setClusteredImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setLoading(true);
      setError(null);

      // Send to backend
      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await axios.post('https://backendforrocksense.onrender.com/api/cluster', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });

        setClusteredImage(response.data.clusteredImage); // This will now be a base64 string
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

  return (
    <div className="min-h-screen bg-gray-50">
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
              <p className="mt-2 text-sm text-gray-600">
                Drag and drop your mineral image here, or click to select a file
              </p>
            )}
          </div>
        </div>

        {error && (
          <div className="mb-8 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded relative">
            {error}
          </div>
        )}

        {/* Processed Output Image */}
        {clusteredImage && (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-lg font-semibold mb-4">Clustered Result</h2>
            <img src={clusteredImage} alt="Clustered" className="w-full rounded" />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
