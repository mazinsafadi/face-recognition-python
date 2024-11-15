// components/ui/alert/alert.jsx
import React from 'react';
import { Alert as AlertIcon, AlertCircle } from 'lucide-react';

const Alert = ({ children, variant = "default", className = "", ...props }) => {
  const baseStyles = "relative w-full rounded-lg border p-4 [&>svg~*]:pl-7 [&>svg+div]:translate-y-[-3px] [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4 [&>svg]:text-foreground";

  const variants = {
    default: "bg-gray-50 text-gray-900 border-gray-200 [&>svg]:text-gray-900",
    destructive: "border-red-500/50 text-red-600 dark:border-red-500 [&>svg]:text-red-600 bg-red-50",
    success: "border-green-500/50 text-green-600 [&>svg]:text-green-600 bg-green-50",
    warning: "border-yellow-500/50 text-yellow-600 [&>svg]:text-yellow-600 bg-yellow-50",
  };

  const icons = {
    default: <AlertIcon className="h-4 w-4" />,
    destructive: <AlertCircle className="h-4 w-4" />,
    success: <AlertIcon className="h-4 w-4" />,
    warning: <AlertCircle className="h-4 w-4" />,
  };

  return (
    <div
      role="alert"
      className={`${baseStyles} ${variants[variant]} ${className}`}
      {...props}
    >
      {icons[variant]}
      <div>{children}</div>
    </div>
  );
};

export const AlertTitle = ({
  className = "",
  ...props
}) => (
  <h5
    className={`mb-1 font-medium leading-none tracking-tight ${className}`}
    {...props}
  />
);

export const AlertDescription = ({
  className = "",
  ...props
}) => (
  <div
    className={`text-sm [&_p]:leading-relaxed ${className}`}
    {...props}
  />
);

// components/ui/alert/index.js
export { Alert, AlertTitle, AlertDescription } from './alert';