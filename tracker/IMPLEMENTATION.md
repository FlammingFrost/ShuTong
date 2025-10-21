# Tracker Implementation Summary

## What was implemented

A simple and flexible Python decorator-based tracker that records function calls and stores data in a SQLite database.

## Key Components

1. **tracker.py** - Main tracker implementation
   - `Tracker` class with decorator functionality
   - SQLite database storage at `./data/tracker/record.db`
   - Value extraction from function arguments, return values, and errors
   - Query methods to retrieve records

2. **example.py** - Usage examples
   - Demonstrates tracking LLM calls, data processing, and error handling
   - Shows how to query records from the database

3. **view_records.py** - Utility script to view tracked records
   - Command-line tool to inspect database contents
   - Supports filtering by name and limiting results

## Usage

### Basic decorator usage:
```python
from tracker import tracker

@tracker(name="llm_gen", value={
    "query": 'args[0].get("user_query")',
    "answer": 'ret[0].get("llm_response")'
})
def get_llm_generation(state):
    return {"llm_response": "Some answer"}
```

### Query records:
```python
from tracker import Tracker

t = Tracker()
records = t.get_records(name="llm_gen", limit=10)
```

### View from command line:
```bash
python -m tracker.view_records --name llm_gen --verbose
```

## Database Schema

```sql
CREATE TABLE records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    function TEXT NOT NULL,
    tracked_values TEXT NOT NULL,  -- JSON string
    status TEXT NOT NULL,
    error TEXT,
    extraction_failed INTEGER DEFAULT 0
);
```

## Features

✅ Decorator-based tracking with `@tracker`
✅ SQLite storage at `./data/tracker/record.db`
✅ Flexible value extraction using Python expressions
✅ Supports tracking `args`, `ret[0]`, and `err`
✅ Handles both success and error cases
✅ Query methods to retrieve records
✅ Command-line utility to view records
✅ Automatically creates database and tables
✅ Thread-safe with per-call connections

## Testing

Run the example:
```bash
python -m tracker.example
```

View records:
```bash
python -m tracker.view_records --limit 5
python -m tracker.view_records --name llm_gen --verbose
```
