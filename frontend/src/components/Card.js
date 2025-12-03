import React from 'react';
import { cn } from '../utils/helpers';

const Card = ({ children, className = '', hover = false, ...props }) => {
  return (
    <div
      className={cn(
        'bg-white rounded-xl shadow-md p-6',
        hover && 'card-hover cursor-pointer',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};

export default Card;
