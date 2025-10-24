# Problem Generator Page

A new feature in the ShuTong Streamlit app that allows users to generate math problems using AI.

## Features

### 🎲 AI-Powered Problem Generation
- Enter any math topic or concept
- Choose difficulty level: Undergraduate, Graduate, Advanced Graduate, Research Level
- Select problem type: Proof, Calculation, Application, Conceptual, Mixed
- Generate problems using GPT-4o with creative temperature (0.8)

### ✏️ Editable Text Area
- Generated problems appear in an editable text area
- Modify the problem as needed
- Full LaTeX support with `$` and `$$` delimiters
- Text is fully copyable

### 🎨 Real-Time Rendered Preview
- See how your problem looks with LaTeX rendering
- Updates automatically when you edit the text
- Beautiful mathematical notation using MathJax
- Converts `\(` `\)` and `\[` `\]` to proper Streamlit format

### 📤 Export to Main Page
- One-click button to use generated problem in Agent QA
- Problem is automatically pre-filled in the Main Page
- Seamless workflow from generation to solving

## How to Use

1. **Navigate to Problem Generator**
   - Click "Problem Generator" in the sidebar

2. **Enter Topic**
   - Type a math topic (e.g., "probability theory", "group theory", "topology")

3. **Configure Options**
   - Select difficulty level
   - Choose problem type

4. **Generate**
   - Click "Generate Problem"
   - Wait for AI to create the problem

5. **Edit & Preview**
   - Edit the generated problem in the text area
   - See real-time rendered preview below
   - Copy the markdown code if needed

6. **Use in Agent QA (Optional)**
   - Click "Use This Problem in Agent QA"
   - Switch to Main Page
   - Problem will be pre-filled and ready to solve

## Example Topics

- **Probability**: "conditional probability and Bayes theorem"
- **Calculus**: "multivariable optimization with constraints"
- **Linear Algebra**: "eigenvalues and spectral decomposition"
- **Abstract Algebra**: "group homomorphisms and normal subgroups"
- **Real Analysis**: "uniform continuity and compactness"
- **Topology**: "connectedness and compactness in metric spaces"
- **Differential Equations**: "boundary value problems for PDEs"

## LaTeX Support

The generator creates problems with proper LaTeX notation:
- Inline math: `$x^2 + y^2 = r^2$`
- Display math: `$$\int_0^\infty e^{-x} dx = 1$$`
- All standard LaTeX commands supported

## Technical Details

- **Model**: GPT-4o
- **Temperature**: 0.8 (creative generation)
- **Prompt Engineering**: Specialized system prompt for mathematical problem generation
- **Output Format**: Markdown with LaTeX
- **Rendering**: Streamlit's markdown with MathJax support

## Benefits

1. **Quick Problem Creation**: Generate problems in seconds
2. **Customizable Difficulty**: Match your learning level
3. **Varied Problem Types**: Different styles of mathematical challenges
4. **Editable Output**: Tweak generated problems to your needs
5. **Seamless Integration**: Direct export to solver pipeline
6. **Visual Preview**: See exactly how the problem will look

Enjoy creating interesting math problems! 🎓
