# ShuTong Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                                 │
│                     http://localhost:3000                            │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ HTTP Requests
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     REACT FRONTEND (Port 3000)                       │
├─────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │
│  │   Page 1      │  │   Page 2      │  │   Page 3      │           │
│  │  Overview     │  │  Problem Gen  │  │  Agent Solver │           │
│  │  Charts       │  │  AI Generate  │  │  Real-time    │           │
│  └───────────────┘  └───────────────┘  └───────────────┘           │
│         │                   │                   │                    │
│         └───────────────────┴───────────────────┘                    │
│                             │                                        │
│                   ┌─────────▼─────────┐                              │
│                   │   API Service     │                              │
│                   │   (api.js)        │                              │
│                   └─────────┬─────────┘                              │
│                             │                                        │
│         ┌──────────────┬────┴────┬──────────────┐                   │
│         │              │         │              │                    │
│   ┌─────▼─────┐  ┌────▼────┐  ┌▼─────┐  ┌────▼────┐                │
│   │LatexRender│  │ Button  │  │ Card │  │ Loading │                │
│   │  KaTeX    │  │ Alert   │  │Layout│  │ Spinner │                │
│   └───────────┘  └─────────┘  └──────┘  └─────────┘                │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ Axios HTTP / SSE
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FLASK API SERVER (Port 8000)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  GET  /api/health               → Health check                       │
│  GET  /api/analysis             → Model performance metrics          │
│  POST /api/generate-problem     → Generate math problem (GPT-4o)     │
│  POST /api/run-agent            → Run agent pipeline                 │
│  POST /api/run-agent-stream     → Stream agent progress (SSE)        │
│  GET  /api/runs                 → Get run history                    │
│  GET  /api/runs/:id             → Get run details                    │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Request Handlers                          │    │
│  │  • CORS enabled                                              │    │
│  │  • JSON responses                                            │    │
│  │  • Error handling                                            │    │
│  │  • Streaming support                                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                      │          │          │
        ┌─────────────┼──────────┼──────────┼─────────────┐
        │             │          │          │             │
        ▼             ▼          ▼          ▼             ▼
┌──────────────┐ ┌─────────┐ ┌──────┐ ┌─────────┐ ┌────────────┐
│ AgentPipeline│ │ Tracker │ │ Eval │ │ ChatGPT │ │   Results  │
│              │ │         │ │Bench │ │  API    │ │ Directory  │
│ solver_agent │ │SQLite DB│ │      │ │         │ │ gpt-*.json │
│ critic_agent │ │         │ │      │ │         │ │            │
└──────────────┘ └─────────┘ └──────┘ └─────────┘ └────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. USER ACTION (Browser)                                            │
│     ↓                                                                 │
│  2. React Component Event Handler                                    │
│     ↓                                                                 │
│  3. API Service Call (Axios)                                         │
│     ↓                                                                 │
│  4. HTTP Request to Flask                                            │
│     ↓                                                                 │
│  5. Flask Route Handler                                              │
│     ↓                                                                 │
│  6. Python Agent/Tracker/Analysis                                    │
│     ↓                                                                 │
│  7. JSON Response / SSE Stream                                       │
│     ↓                                                                 │
│  8. React State Update                                               │
│     ↓                                                                 │
│  9. UI Re-render with Results                                        │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                    REAL-TIME UPDATES (Page 3)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  User clicks "Run Agent"                                             │
│       ↓                                                               │
│  POST /api/run-agent-stream                                          │
│       ↓                                                               │
│  Flask opens SSE connection                                          │
│       ↓                                                               │
│  AgentPipeline.run() executes                                        │
│       ↓                                                               │
│  ┌─────────────────────────────────────────┐                         │
│  │ Stream events as they happen:           │                         │
│  │  • Stage: initializing                  │                         │
│  │  • Stage: initial → solution_step       │                         │
│  │  • Stage: critique → critique_result    │                         │
│  │  • Stage: refine → solution_step        │                         │
│  │  • Stage: completed → final_result      │                         │
│  └─────────────────────────────────────────┘                         │
│       ↓                                                               │
│  React receives events via EventSource                               │
│       ↓                                                               │
│  UI updates in real-time                                             │
│   • Progress bar animates                                            │
│   • Steps appear one by one                                          │
│   • Critiques show after each step                                   │
│   • Auto-scroll to latest                                            │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                      TECHNOLOGY STACK                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Frontend (React)                      Backend (Python)              │
│  ├─ React 18                          ├─ Flask 3.0                   │
│  ├─ React Router 6                    ├─ Flask-CORS                  │
│  ├─ Recharts (Charts)                 ├─ LangChain                   │
│  ├─ Axios (HTTP)                      ├─ OpenAI API                  │
│  ├─ KaTeX (LaTeX)                     ├─ SQLite (Tracker)            │
│  ├─ TailwindCSS (Styling)             └─ [existing dependencies]     │
│  ├─ Lucide React (Icons)                                             │
│  └─ PostCSS (Processing)                                             │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                      FILE STRUCTURE                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  project-root/                                                       │
│  ├── frontend/                  ← React app                          │
│  │   ├── public/                ← Static files                       │
│  │   ├── src/                   ← Source code                        │
│  │   │   ├── components/        ← Reusable UI                        │
│  │   │   ├── pages/             ← Main pages                         │
│  │   │   ├── services/          ← API client                         │
│  │   │   └── utils/             ← Helpers                            │
│  │   └── package.json           ← Dependencies                       │
│  │                                                                    │
│  ├── agent_sys/                 ← Agent pipeline                     │
│  ├── critic_agent/              ← Critic system                      │
│  ├── solver_agent/              ← Solver system                      │
│  ├── tracker/                   ← Run tracking                       │
│  ├── eval/                      ← Evaluation tools                   │
│  ├── results/                   ← Model results                      │
│  ├── data/                      ← Tracker database                   │
│  │                                                                    │
│  ├── api_server.py              ← Flask API                          │
│  ├── app.py                     ← Streamlit app                      │
│  ├── start_frontend.sh          ← Startup script                     │
│  └── [documentation files]                                           │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Architectural Decisions

### 1. **Separation of Concerns**
   - Frontend: Pure presentation and user interaction
   - API: Business logic coordination
   - Python modules: Core functionality (agents, tracking, analysis)

### 2. **Real-time Communication**
   - Server-Sent Events (SSE) for one-way streaming
   - Alternative: WebSockets (bi-directional, more complex)
   - Fallback: Polling for environments without SSE

### 3. **State Management**
   - Local component state for simple UI
   - Session storage for cross-page data (generated problems)
   - No global state management needed yet (could add Redux/Zustand)

### 4. **API Design**
   - RESTful endpoints for CRUD operations
   - Streaming endpoint for real-time updates
   - CORS enabled for development
   - JSON responses throughout

### 5. **Styling Approach**
   - TailwindCSS for utility-first styling
   - Custom theme with primary/accent colors
   - Component-based styling (no global CSS pollution)
   - Responsive design with mobile-first approach

### 6. **Error Handling**
   - Try-catch in all async operations
   - User-friendly error messages
   - API error responses with status codes
   - Loading states for async actions

### 7. **Performance**
   - Lazy loading potential (not yet implemented)
   - Debounced inputs for search/filter
   - Memoization for expensive computations
   - Code splitting with React Router

## Scalability Considerations

### Current Setup (Development)
- ✅ Single Flask instance
- ✅ Local SQLite database
- ✅ Direct API calls
- ✅ Simple deployment

### Production Ready (Future)
- Add nginx reverse proxy
- Use gunicorn for Flask
- Add Redis for caching
- PostgreSQL for production DB
- Load balancing for multiple API instances
- CDN for static assets
- Environment-based configuration

## Security Notes

### Current (Development)
- CORS wide open for development
- No authentication
- Direct API access

### Production Should Add
- API key authentication
- Rate limiting
- HTTPS/TLS
- CORS restricted to frontend domain
- Input validation and sanitization
- SQL injection prevention (already using ORM)
- XSS protection (React handles this)
