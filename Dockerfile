#using a base image with Python 3.9
FROM python:3.9-slim

#installing necessary dependencies for building and running the application
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#setting the working directory inside the containerto /app
WORKDIR /app

#copying the requirements file and installing the dependencies listed in it
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copying the rest of the project files
COPY . .

#specifying the command to run when the container starts, which in this case is to start a Python interpreter
CMD ["python"]