"""
Tracker decorator for recording function calls with customizable value extraction.
"""

import json
import sqlite3
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class Tracker:
    """Tracker class for recording function execution data."""
    
    def __init__(self, data_dir: str = "./data/tracker"):
        """
        Initialize the Tracker.
        
        Args:
            data_dir: Directory to store tracker data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "record.db"
        self.current_run_id: Optional[str] = None
        self._init_database()
    
    def set_run_id(self, run_id: str) -> None:
        """
        Set the current run ID for tracking.
        
        Args:
            run_id: The run identifier to use for subsequent tracked calls
        """
        self.current_run_id = run_id
    
    def clear_run_id(self) -> None:
        """Clear the current run ID."""
        self.current_run_id = None
    
    def _init_database(self) -> None:
        """Initialize the SQLite database with the required schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='records'
        """)
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # Check if run_id column exists (for migration)
            cursor.execute("PRAGMA table_info(records)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'run_id' not in columns:
                # Migrate: add run_id column
                cursor.execute("ALTER TABLE records ADD COLUMN run_id TEXT")
                conn.commit()
        else:
            # Create new table
            cursor.execute("""
                CREATE TABLE records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    function TEXT NOT NULL,
                    tracked_values TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    extraction_failed INTEGER DEFAULT 0
                )
            """)
            conn.commit()
        
        # Create index on run_id for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_run_id ON records(run_id)
        """)
        
        conn.commit()
        conn.close()
    
    def _evaluate_value_expression(self, expression: str, args: tuple, ret: Any, err: Optional[Exception]) -> Any:
        """
        Evaluate a value expression like 'args[0].get("key")' or 'ret[0].get("answer")'.
        
        Args:
            expression: The expression to evaluate
            args: Function arguments
            ret: Function return value
            err: Exception if any occurred
            
        Returns:
            The evaluated value
            
        Raises:
            Any exception that occurs during evaluation
        """
        # Create a safe evaluation context with allowed builtins
        safe_builtins = {
            'len': len,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'enumerate': enumerate,
            'range': range,
            'zip': zip,
            'True': True,
            'False': False,
            'None': None,
        }
        
        context = {
            'args': args,
            'ret': ret,
            'err': err,
        }
        
        try:
            # Evaluate the expression with safe builtins
            result = eval(expression, {"__builtins__": safe_builtins}, context)
            return result
        except Exception as e:
            # Re-raise the exception as specified in requirements
            raise e
    
    def _extract_values(self, value_spec: Dict[str, str], args: tuple, ret: Any, err: Optional[Exception]) -> Dict[str, Any]:
        """
        Extract values based on the value specification.
        
        Args:
            value_spec: Dictionary mapping field names to expressions
            args: Function arguments
            ret: Function return value (wrapped in list for indexing)
            err: Exception if any occurred
            
        Returns:
            Dictionary of extracted values
        """
        extracted = {}
        
        for key, expression in value_spec.items():
            try:
                extracted[key] = self._evaluate_value_expression(expression, args, ret, err)
            except Exception as e:
                # Re-raise as specified in requirements
                raise e
        
        return extracted
    
    def _save_record(self, name: str, record: Dict[str, Any]) -> None:
        """
        Save a record to the SQLite database.
        
        Args:
            name: Name of the tracked function
            record: Record data to save
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO records (run_id, name, timestamp, function, tracked_values, status, error, extraction_failed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.get("run_id"),
            record["name"],
            record["timestamp"],
            record["function"],
            json.dumps(record.get("values", {}), ensure_ascii=False),
            record["status"],
            record.get("error"),
            1 if record.get("extraction_failed", False) else 0
        ))
        
        conn.commit()
        conn.close()
    
    def get_records(self, name: Optional[str] = None, limit: Optional[int] = None, run_id: Optional[str] = None) -> list:
        """
        Retrieve records from the database.
        
        Args:
            name: Filter by function name (optional)
            limit: Maximum number of records to return (optional)
            run_id: Filter by run ID (optional)
            
        Returns:
            List of record dictionaries
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM records"
        params = []
        conditions = []
        
        if name:
            conditions.append("name = ?")
            params.append(name)
        
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        records = []
        for row in rows:
            record = {
                "id": row["id"],
                "run_id": row["run_id"],
                "name": row["name"],
                "timestamp": row["timestamp"],
                "function": row["function"],
                "values": json.loads(row["tracked_values"]) if row["tracked_values"] else {},
                "status": row["status"],
                "error": row["error"],
                "extraction_failed": bool(row["extraction_failed"])
            }
            records.append(record)
        
        conn.close()
        return records
    
    def get_all_run_ids(self) -> list:
        """
        Get all distinct run IDs from the database, ordered by most recent.
        
        Returns:
            List of dictionaries with run_id, timestamp, and problem information
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get unique run_ids with their first timestamp and problem info
        query = """
            SELECT 
                run_id,
                MIN(timestamp) as first_timestamp,
                MAX(timestamp) as last_timestamp
            FROM records
            WHERE run_id IS NOT NULL
            GROUP BY run_id
            ORDER BY MIN(timestamp) DESC
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        run_ids = []
        for row in rows:
            # Try to get the math problem for this run
            cursor.execute(
                "SELECT tracked_values FROM records WHERE run_id = ? AND name = 'pipeline_run' LIMIT 1",
                (row["run_id"],)
            )
            problem_row = cursor.fetchone()
            
            problem_text = None
            if problem_row:
                try:
                    values = json.loads(problem_row["tracked_values"])
                    problem_text = values.get("math_problem", "")
                except:
                    pass
            
            run_ids.append({
                "run_id": row["run_id"],
                "first_timestamp": row["first_timestamp"],
                "last_timestamp": row["last_timestamp"],
                "problem": problem_text
            })
        
        conn.close()
        return run_ids
    
    def track(self, name: str, value: Dict[str, str]) -> Callable:
        """
        Decorator to track function calls.
        
        Args:
            name: Name for this tracked function
            value: Dictionary mapping field names to value expressions
                   Expressions can reference:
                   - args: function arguments tuple
                   - ret: function return value (always ret[0] for the actual return)
                         If function returns tuple (a,b), use ret[0][0] for a, ret[0][1] for b
                   - err: exception if any occurred
                   
        Example:
            @tracker.track(name="llm_gen", value={
                "query": 'args[0].get("user_query")',
                "answer": 'ret[0].get("llm_response")'
            })
            def get_llm_generation(state):
                # llm generation
                return llm.result.response
                
            @tracker.track(name="solve", value={
                "solution": 'ret[0][0]',  # First element of tuple
                "steps": 'ret[0][1]'      # Second element of tuple
            })
            def solve(problem):
                solution = "..."
                steps = [...]
                return solution, steps
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                ret = None
                err = None
                
                try:
                    # Execute the function
                    result = func(*args, **kwargs)
                    ret = [result]  # Wrap in list so ret[0] is the actual return value
                    
                    # Extract values
                    extracted_values = self._extract_values(value, args, ret, err)
                    
                    # Create record
                    record = {
                        "run_id": self.current_run_id,
                        "name": name,
                        "timestamp": datetime.now().isoformat(),
                        "function": func.__name__,
                        "values": extracted_values,
                        "status": "success"
                    }
                    
                    # Save record
                    self._save_record(name, record)
                    
                    return result
                    
                except Exception as e:
                    err = e
                    
                    # Try to extract values even on error (some might still work)
                    try:
                        extracted_values = self._extract_values(value, args, ret, err)
                        record = {
                            "run_id": self.current_run_id,
                            "name": name,
                            "timestamp": datetime.now().isoformat(),
                            "function": func.__name__,
                            "values": extracted_values,
                            "status": "error",
                            "error": str(e)
                        }
                        self._save_record(name, record)
                    except:
                        # If extraction fails during error, just record the error
                        record = {
                            "run_id": self.current_run_id,
                            "name": name,
                            "timestamp": datetime.now().isoformat(),
                            "function": func.__name__,
                            "status": "error",
                            "error": str(e),
                            "extraction_failed": True
                        }
                        self._save_record(name, record)
                    
                    # Re-raise the original exception
                    raise e
            
            return wrapper
        return decorator


# Create a default tracker instance
_default_tracker = Tracker()


def tracker(name: str, value: Dict[str, str]) -> Callable:
    """
    Convenience decorator using the default tracker instance.
    
    Args:
        name: Name for this tracked function
        value: Dictionary mapping field names to value expressions
               
    Example:
        @tracker(name="llm_gen", value={
            "query": 'args[0].get("user_query")',
            "answer": 'ret[0].get("llm_response")'
        })
        def get_llm_generation(state):
            # llm generation
            return llm.result.response
    """
    return _default_tracker.track(name, value)
