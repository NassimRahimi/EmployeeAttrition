FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files to the container
COPY . .

# Expose ports for the Streamlit app and FastAPI
EXPOSE 8501 8000

# Define environment variables for Streamlit
ENV PYTHONPATH=.

# Start both FastAPI and Streamlit using a custom script or command
CMD ["sh", "-c", "uvicorn fast_api_main:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.port=8501 --server.enableCORS false"]
