// components/auth/registration-form.jsx
import React, { useState } from 'react';
import { Button } from '../button';
import { Input } from '../input';
import { Alert } from '../alert';
import { Card, CardHeader, CardTitle, CardContent } from '../card';
import FaceCapture from './face-capture';

const RegistrationForm = ({ onRegister, onCancel }) => {
  const [name, setName] = useState('');
  const [showCamera, setShowCamera] = useState(false);
  const [error, setError] = useState('');

  const handlePhotoCapture = async (blob) => {
    if (!name.trim()) {
      setError('Please enter your name');
      return;
    }

    try {
      const formData = new FormData();
      formData.append('photo', blob);
      formData.append('name', name);
      await onRegister(formData);
    } catch (err) {
      setError('Registration failed');
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
        <CardTitle>Register New Face</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            {error}
          </Alert>
        )}
        <Input
          type="text"
          placeholder="Enter your name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <div className="flex space-x-4">
          <Button
            onClick={() => setShowCamera(true)}
            className="flex-1"
          >
            Start Camera
          </Button>
          <Button
            variant="outline"
            onClick={onCancel}
          >
            Back to Login
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default RegistrationForm;