// components/ui/alert/alert.jsx
import React from 'react';
import { AlertTriangle, AlertCircle } from 'lucide-react';

const Alert = ({ children, variant = "default", className = "", ...props }) => {
  const baseStyles = "relative w-full rounded-lg border p-4 [&>svg~*]:pl-7 [&>svg+div]:translate-y-[-3px] [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-4 [&>svg]:text-foreground";

  const variants = {
    default: "bg-gray-50 text-gray-900 border-gray-200 [&>svg]:text-gray-900",
    destructive: "border-red-500/50 text-red-600 dark:border-red-500 [&>svg]:text-red-600 bg-red-50",
    success: "border-green-500/50 text-green-600 [&>svg]:text-green-600 bg-green-50",
    warning: "border-yellow-500/50 text-yellow-600 [&>svg]:text-yellow-600 bg-yellow-50",
  };

  const icons = {
    default: <AlertTriangle className="h-4 w-4" />,
    destructive: <AlertCircle className="h-4 w-4" />,
    success: <AlertTriangle className="h-4 w-4" />,
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

Alert.displayName = "Alert";

export const AlertTitle = React.forwardRef(({ className = "", ...props }, ref) => (
  <h5
    ref={ref}
    className={`mb-1 font-medium leading-none tracking-tight ${className}`}
    {...props}
  />
));
AlertTitle.displayName = "AlertTitle";

export const AlertDescription = React.forwardRef(({ className = "", ...props }, ref) => (
  <div
    ref={ref}
    className={`text-sm [&_p]:leading-relaxed ${className}`}
    {...props}
  />
));
AlertDescription.displayName = "AlertDescription";

export { Alert };