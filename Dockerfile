# Base image: Python with anaconda for scientific computing
FROM continuumio/anaconda3:latest

# Set working directory
WORKDIR /app

# Copy the environment file and install dependencies
COPY environment.yml /app/environment.yml
RUN conda env create -f environment.yml && conda clean --all

# Activate the environment
SHELL ["conda", "run", "-n", "hybrid_chess_ai", "/bin/bash", "-c"]

# Copy the rest of the codebase
COPY . /app

# Expose a port for the GUI (if applicable)
EXPOSE 5000

# Default command to run when the container starts
CMD ["conda", "run", "--no-capture-output", "-n", "hybrid_chess_ai", "python", "src/gui/main.py"]