import React from 'react';
import { AlertCircle, CheckCircle, Info, XCircle } from 'lucide-react';

const Alert = ({ type = 'info', title, message, className = '' }) => {
  const config = {
    info: {
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200',
      textColor: 'text-blue-800',
      icon: Info,
      iconColor: 'text-blue-500',
    },
    success: {
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200',
      textColor: 'text-green-800',
      icon: CheckCircle,
      iconColor: 'text-green-500',
    },
    warning: {
      bgColor: 'bg-yellow-50',
      borderColor: 'border-yellow-200',
      textColor: 'text-yellow-800',
      icon: AlertCircle,
      iconColor: 'text-yellow-500',
    },
    error: {
      bgColor: 'bg-red-50',
      borderColor: 'border-red-200',
      textColor: 'text-red-800',
      icon: XCircle,
      iconColor: 'text-red-500',
    },
  };

  const { bgColor, borderColor, textColor, icon: Icon, iconColor } = config[type];

  return (
    <div className={`${bgColor} ${borderColor} border rounded-lg p-4 ${className}`}>
      <div className="flex items-start space-x-3">
        <Icon className={`${iconColor} flex-shrink-0 mt-0.5`} size={20} />
        <div className="flex-1">
          {title && <h4 className={`${textColor} font-semibold mb-1`}>{title}</h4>}
          {message && <p className={`${textColor} text-sm`}>{message}</p>}
        </div>
      </div>
    </div>
  );
};

export default Alert;
