# ShuTong React Frontend

A modern, interactive React.js frontend for the ShuTong Math Agent System.

## Features

### 📊 Page 1: Overview & Efficiency Charts
- Interactive model comparison charts
- Switch between metrics:
  - **Exact Match Metrics**: Accuracy, Precision, Recall, F1-Score
  - **Correct Match Metrics**: Accuracy, Precision, Recall, F1-Score
  - **Cost Efficiency**: Cost per exact match, Cost per correct match
- Compare performance across different models (GPT-5.1, GPT-5 Mini, GPT-5 Nano, GPT-4o Mini)
- Summary statistics cards for each model

### ✍️ Page 2: Problem Generator
- Generate math problems using AI based on:
  - Topic/knowledge point
  - Difficulty level (Undergraduate, Graduate, Advanced Graduate, Research Level)
  - Problem type (Proof, Calculation, Application, Conceptual, Mixed)
- Real-time LaTeX rendering
- Editable problem text
- Quick example topics
- Send generated problems directly to the solver

### 🤖 Page 3: Agent Solver
- Real-time agent execution with progress tracking
- Step-by-step solution display with LaTeX support
- Live critique analysis for each step:
  - Logic correctness
  - Calculation accuracy
  - Knowledge points
- Visual indicators for errors and corrections
- Refinement iteration tracking
- Final solution summary with metrics

## Technology Stack

- **React 18** - UI framework
- **React Router** - Navigation
- **Recharts** - Interactive charts
- **TailwindCSS** - Styling
- **KaTeX** - LaTeX math rendering
- **Axios** - API client
- **Lucide React** - Icons

## Prerequisites

- Node.js 16.x or higher
- npm or yarn
- Python backend API running (see Backend Setup below)

## Installation

### 1. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 2. Set Up Backend API

Install Python API dependencies:

```bash
# From project root
uv pip install flask flask-cors
# Or use requirements file
uv pip install -r requirements-api.txt
```

### 3. Configure API Endpoint (Optional)

Create `.env` file in the `frontend` directory:

```env
REACT_APP_API_URL=http://localhost:8000
```

Default is `http://localhost:8000` if not specified.

## Running the Application

### Start the Backend API Server

From the project root directory:

```bash
uv run python api_server.py
```

The API will start on `http://localhost:8000`

### Start the React Development Server

In the `frontend` directory:

```bash
npm start
```

The app will open at `http://localhost:3000`

## Building for Production

```bash
cd frontend
npm run build
```

This creates an optimized production build in the `build` directory.

## Project Structure

```
frontend/
├── public/
│   ├── index.html          # HTML template
│   └── manifest.json       # PWA manifest
├── src/
│   ├── components/         # Reusable components
│   │   ├── Layout.js       # Main layout with navigation
│   │   ├── Card.js         # Card component
│   │   ├── Button.js       # Button component
│   │   ├── Alert.js        # Alert component
│   │   ├── LoadingSpinner.js  # Loading indicator
│   │   └── LatexRenderer.js   # LaTeX rendering
│   ├── pages/              # Page components
│   │   ├── Overview.js     # Page 1: Charts and analytics
│   │   ├── ProblemGenerator.js  # Page 2: Problem generation
│   │   └── AgentSolver.js  # Page 3: Agent solver
│   ├── services/           # API services
│   │   └── api.js          # API client and methods
│   ├── utils/              # Utility functions
│   │   └── helpers.js      # Helper functions
│   ├── App.js              # Main app component
│   ├── index.js            # Entry point
│   └── index.css           # Global styles
├── package.json            # Dependencies and scripts
├── tailwind.config.js      # TailwindCSS configuration
└── postcss.config.js       # PostCSS configuration
```

## API Endpoints

The React frontend communicates with the following API endpoints:

### Analysis Data
```
GET /api/analysis
Returns model performance metrics for charts
```

### Problem Generation
```
POST /api/generate-problem
Body: { topic, difficulty, problemType }
Generates a math problem using LLM
```

### Agent Execution
```
POST /api/run-agent
Body: { math_problem, max_iterations }
Runs the agent pipeline on a problem
```

### Streaming Agent (Real-time)
```
POST /api/run-agent-stream
Body: { math_problem, max_iterations }
Streams agent progress via Server-Sent Events
```

### Run History
```
GET /api/runs
Returns list of all runs

GET /api/runs/:runId
Returns detailed records for a specific run
```

## Features in Detail

### Real-time Updates

The Agent Solver page uses either:
- **Server-Sent Events (SSE)** for real-time streaming from the backend
- **Polling** as a fallback mechanism
- **Simulated progress** for demo mode

Progress updates include:
- Current stage (initializing, generating, critiquing, refining)
- Solution steps as they're generated
- Critique analysis for each step
- Final results and metrics

### LaTeX Support

All math content supports LaTeX notation:
- Inline math: `$x^2 + y^2 = z^2$`
- Display math: `$$\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$`

LaTeX is rendered using KaTeX for fast, high-quality display.

### Responsive Design

The interface is fully responsive and works on:
- Desktop (optimized experience)
- Tablet (adjusted layouts)
- Mobile (touch-friendly interface)

## Customization

### Changing Colors

Edit `tailwind.config.js`:

```javascript
theme: {
  extend: {
    colors: {
      primary: { /* your colors */ },
      accent: { /* your colors */ },
    }
  }
}
```

### Adding New Models

Update the mock data in `Overview.js` or modify the API to return data for new models.

### Modifying Chart Types

The app uses Recharts. You can easily switch between:
- Bar charts
- Line charts  
- Area charts
- Composed charts

See [Recharts documentation](https://recharts.org/) for more options.

## Troubleshooting

### API Connection Issues

1. Verify the backend API is running on `http://localhost:8000`
2. Check CORS is enabled in the Flask app
3. Verify the `REACT_APP_API_URL` in `.env`

### LaTeX Rendering Issues

1. Ensure KaTeX CSS is loaded in `public/index.html`
2. Check LaTeX syntax is correct
3. Use `convertLatex()` helper for legacy delimiters

### Build Errors

1. Clear node_modules: `rm -rf node_modules && npm install`
2. Clear cache: `npm cache clean --force`
3. Update Node.js to latest LTS version

## Development Tips

### Hot Reload

The development server supports hot module replacement. Changes to components will update without full page reload.

### Debug Mode

Add to `.env`:
```
REACT_APP_DEBUG=true
```

### API Mocking

For development without backend, the pages include mock data that can be used by commenting out API calls.

## Contributing

When adding new features:

1. Create reusable components in `components/`
2. Add new pages in `pages/`
3. Update routing in `App.js`
4. Add API methods to `services/api.js`
5. Use TailwindCSS for styling
6. Follow existing naming conventions

## Performance Optimization

The app includes:
- Code splitting with React.lazy (can be added)
- Optimized re-renders with React.memo (can be added)
- Debounced inputs for search/filter
- Lazy loading for images
- Production build optimization

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

## License

See main project LICENSE file.

## Support

For issues or questions:
1. Check this README
2. Review the API server logs
3. Check browser console for errors
4. Verify all dependencies are installed

## Acknowledgments

- Built with React and modern web technologies
- Icons by Lucide React
- Charts by Recharts
- Math rendering by KaTeX
- Styling by TailwindCSS
