// src/FaceAuth.jsx
import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/card';
import { Alert } from './components/ui/alert';
import LoginForm from './components/auth/login-form';
import RegistrationForm from './components/auth/registration-form';

const FaceAuth = () => {
  const [isRegistering, setIsRegistering] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const handleLogin = async (formData) => {
    try {
      const response = await fetch(`${API_URL}/api/signin`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setMessage(data.message);
      } else {
        setError(data.detail || 'Authentication failed');
      }
    } catch (err) {
      setError('Failed to authenticate');
    }
  };

  const handleRegister = async (formData) => {
    try {
      const response = await fetch(`${API_URL}/api/signup`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setMessage(`Hello there, ${formData.get('name')}!`);
      } else {
        setError(data.detail || 'Registration failed');
      }
    } catch (err) {
      setError('Failed to register');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        {message ? (
          <Card>
            <CardContent className="p-6">
              <Alert className="bg-green-50">
                {message}
              </Alert>
            </CardContent>
          </Card>
        ) : (
          <>
            {isRegistering ? (
              <RegistrationForm
                onRegister={handleRegister}
                onCancel={() => setIsRegistering(false)}
              />
            ) : (
              <LoginForm
                onLogin={handleLogin}
                onRegisterClick={() => setIsRegistering(true)}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default FaceAuth;