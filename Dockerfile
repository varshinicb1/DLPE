# Use official Python runtime as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirement file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# If requirements.txt is missing, we install manually as fallback in this instruction
RUN pip install --no-cache-dir streamlit pandas numpy xgboost scikit-learn networkx matplotlib seaborn plotly pyvis streamlit-agraph requests

# Copy source code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
