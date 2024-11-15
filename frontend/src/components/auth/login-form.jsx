// components/auth/login-form.jsx
import React, { useState } from 'react';
import { Button } from '../button';
import { Alert } from '../alert';
import { Card, CardHeader, CardTitle, CardContent } from '../card';
import FaceCapture from './face-capture';

const LoginForm = ({ onLogin, onRegisterClick }) => {
  const [error, setError] = useState('');
  const [showCamera, setShowCamera] = useState(false);

  const handlePhotoCapture = async (blob) => {
    try {
      const formData = new FormData();
      formData.append('photo', blob);
      await onLogin(formData);
    } catch (err) {
      setError('Authentication failed');
    }
  };

  if (showCamera) {
    return (
      <FaceCapture
        onCapture={handlePhotoCapture}
        onCancel={() => setShowCamera(false)}
      />
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Face Recognition Login</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            {error}
          </Alert>
        )}
        <div className="flex flex-col space-y-4">
          <Button
            onClick={() => setShowCamera(true)}
            className="w-full"
          >
            Sign In with Face
          </Button>
          <Button
            variant="outline"
            onClick={onRegisterClick}
            className="w-full"
          >
            Register New Face
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default LoginForm;