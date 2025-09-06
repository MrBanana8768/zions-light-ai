# Zion's Light AI Service

This repo is designed to provide a containerized AI stack utilizing Open WebUI as the front end, llama.cpp for serving access to the LLM/AI tools and a NGINX reverse proxy to allow for API calls to the llama.cpp OpenAI server in a secure manner.

## Running the application

Copy the .env.example folder to .env and follow the instructions contained there in.

Running the following command will instantiate the instance with the defaults provided.

``` docker-compose up -d ```

``` docker-compose --profile production up -d ``` for production builds
: 
