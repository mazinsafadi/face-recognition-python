// components/auth/face-capture.jsx
import React, { useRef, useEffect } from 'react';
import { Button } from '../ui/button';
import { Card, CardContent } from '../ui/card';
import { Camera } from 'lucide-react';

const FaceCapture = ({ onCapture, onCancel }) => {
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      streamRef.current = stream;
    } catch (err) {
      console.error('Failed to access camera:', err);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
  };

  const handleCapture = async () => {
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    canvas.toBlob(blob => {
      onCapture(blob);
      stopCamera();
    }, 'image/jpeg');
  };

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardContent className="p-4">
        <div className="relative">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full h-64 object-cover rounded-lg mb-4"
          />
          <Camera className="absolute top-4 right-4 text-white" size={24} />
        </div>
        <div className="flex space-x-4">
          <Button onClick={handleCapture} className="flex-1">
            Capture Photo
          </Button>
          <Button variant="outline" onClick={() => {
            stopCamera();
            onCancel();
          }}>
            Cancel
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default FaceCapture;