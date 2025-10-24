# Quick Start Guide - ShuTong Frontend

## 🚀 Starting the App

1. **Activate the conda environment:**
   ```bash
   conda activate shutong
   ```

2. **Navigate to the project directory:**
   ```bash
   cd /Users/hugo/Documents/Project/2025-10-21-ShuTong
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to [http://localhost:8501](http://localhost:8501)

## 📝 Using the App

### Main Page - Agent QA

1. **Enter your math problem** in the text area
   - Supports LaTeX math: use `$...$` for inline math or `$$...$$` for display math
   - Example: `Prove that $\int_0^\infty e^{-x} dx = 1$`

2. **Adjust settings** (optional)
   - Set max refinement iterations (1-5)

3. **Click "Run Agent Pipeline"**
   - The agent will generate a step-by-step solution
   - Each step will be critiqued for logic and calculation correctness
   - Solutions may be refined based on feedback

4. **Review results:**
   - **Metrics:** View summary statistics
   - **Final Solution:** See the complete solution with LaTeX rendering
   - **Step-by-Step Analysis:** Expand each step to see:
     - Step content with math notation
     - Logic critique (correct/incorrect)
     - Calculation critique (correct/incorrect)
     - Knowledge points identified
   - **Knowledge Points Summary:** All concepts used in the solution
   - **Remaining Issues:** Any problems that couldn't be resolved

### Records Page - View History

1. **Select a run** from the dropdown
   - Shows timestamp and problem preview
   - Organized by most recent first

2. **View run details:**
   - **Run Summary:** Iterations, knowledge points, problem text
   - **Tracked Operations:** All function calls with details
     - Expand operation types to see individual calls
     - View tracked values for each operation
     - See timestamps and status
   - **Reconstructed Solution Path:**
     - Initial solution with all steps
     - Each refinement iteration with feedbacks
     - Final refined solution

## 💡 Tips

- **LaTeX Support:** 
  - The app automatically converts `\(...\)` to `$...$`
  - And `\[...\]` to `$$...$$`
  - So both notations work!

- **Step Critiques:**
  - Green ✅ means correct
  - Red ❌ means issues found
  - Expand steps to see detailed feedback

- **Long Running Times:**
  - Complex problems may take several minutes
  - The app shows a spinner while processing
  - Don't refresh the page while running

- **Database:**
  - All runs are saved to `./data/tracker/record.db`
  - Records persist across sessions
  - You can query the database directly if needed

## 🐛 Troubleshooting

**App won't start:**
```bash
# Kill any existing streamlit processes
pkill -f "streamlit run"
# Then start fresh
streamlit run app.py
```

**No runs showing in Records:**
- Run the agent pipeline at least once on the Main page
- Check that `./data/tracker/record.db` exists

**LaTeX not rendering:**
- Make sure you're using `$` or `$$` delimiters
- Check that your LaTeX syntax is correct
- Try refreshing the browser

## 📚 Example Problems

**Simple calculus:**
```
Find the derivative of f(x) = x³ + 2x² - 5x + 3 at x = 2
```

**Probability:**
```
Prove that for any continuous random variable X with CDF F(x), 
the random variable Y = F(X) follows a uniform distribution on [0,1].
```

**Complex proof:**
```
If X_i are independently distributed according to Γ(α_i, b), 
show that ∑X_i is distributed as Γ(∑α_i, b).
```

Enjoy using ShuTong! 🎓
