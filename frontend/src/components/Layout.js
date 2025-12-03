import React from 'react';
import { NavLink } from 'react-router-dom';
import { BarChart3, PenTool, Sparkles } from 'lucide-react';

const Layout = ({ children }) => {
  const navItems = [
    { to: '/', label: 'Overview', icon: BarChart3 },
    { to: '/generator', label: 'Problem Generator', icon: PenTool },
    { to: '/solver', label: 'Agent Solver', icon: Sparkles },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            {/* Logo */}
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-accent-500 rounded-lg flex items-center justify-center">
                <span className="text-white text-xl font-bold">🧮</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold gradient-text">ShuTong</h1>
                <p className="text-xs text-gray-500">Math Agent System</p>
              </div>
            </div>

            {/* Navigation */}
            <nav className="hidden md:flex space-x-1">
              {navItems.map(({ to, label, icon: Icon }) => (
                <NavLink
                  key={to}
                  to={to}
                  className={({ isActive }) =>
                    `flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                      isActive
                        ? 'bg-primary-500 text-white shadow-lg'
                        : 'text-gray-600 hover:bg-gray-100'
                    }`
                  }
                >
                  <Icon size={18} />
                  <span className="font-medium">{label}</span>
                </NavLink>
              ))}
            </nav>
          </div>

          {/* Mobile Navigation */}
          <nav className="md:hidden flex space-x-1 pb-3 overflow-x-auto">
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `flex items-center space-x-2 px-3 py-2 rounded-lg transition-all duration-200 whitespace-nowrap ${
                    isActive
                      ? 'bg-primary-500 text-white shadow-lg'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`
                }
              >
                <Icon size={16} />
                <span className="text-sm font-medium">{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="text-sm text-gray-600">
              <p className="font-medium">Built with React, TailwindCSS, and GPT-4o</p>
              <p className="text-xs text-gray-500 mt-1">
                An AI-powered math problem solver with solver and critic agents
              </p>
            </div>
            <div className="text-xs text-gray-500">
              © 2025 ShuTong. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
