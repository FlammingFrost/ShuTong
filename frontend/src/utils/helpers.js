import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Merge Tailwind CSS classes with clsx
 */
export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

/**
 * Convert LaTeX delimiters from \( \) and \[ \] to $ and $$
 */
export function convertLatex(text) {
  if (typeof text !== 'string') return text;
  
  // Convert \[ \] to $$
  text = text.replace(/\\\[/g, '$$');
  text = text.replace(/\\\]/g, '$$');
  
  // Convert \( \) to $
  text = text.replace(/\\\(/g, '$');
  text = text.replace(/\\\)/g, '$');
  
  return text;
}

/**
 * Format number with commas
 */
export function formatNumber(num) {
  return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

/**
 * Format percentage
 */
export function formatPercentage(value, decimals = 2) {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format currency
 */
export function formatCurrency(value, decimals = 4) {
  return `$${value.toFixed(decimals)}`;
}

/**
 * Truncate text with ellipsis
 */
export function truncateText(text, maxLength = 100) {
  if (!text || text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
}

/**
 * Get model display name
 */
export function getModelDisplayName(modelKey) {
  const modelNames = {
    'gpt-4o-mini': 'GPT-4o Mini',
    'gpt-5-nano': 'GPT-5 Nano',
    'gpt-5-mini': 'GPT-5 Mini',
    'gpt-5.1': 'GPT-5.1',
    'gpt-5.1-2025-11-13': 'GPT-5.1',
  };
  return modelNames[modelKey] || modelKey;
}

/**
 * Get model color for charts
 */
export function getModelColor(modelKey) {
  const colors = {
    'gpt-4o-mini': '#3b82f6',
    'gpt-5-nano': '#10b981',
    'gpt-5-mini': '#f59e0b',
    'gpt-5.1': '#ef4444',
    'gpt-5.1-2025-11-13': '#ef4444',
  };
  return colors[modelKey] || '#6b7280';
}

/**
 * Debounce function
 */
export function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

/**
 * Sleep/delay utility
 */
export function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
