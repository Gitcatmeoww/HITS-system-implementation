# Semantic Dataset Search Engine

## Getting Started

To run the Semantic Dataset Search Engine locally, follow these steps.

### Prerequisites

Before setting up the project, ensure you have the following installed:

- Node.js and npm
- Python 3.7 or higher
- PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) extension
- An Azure OpenAI API key

### Installation

#### Cloning the Repository

- Start by cloning the repository to your local machine:

```bash
git clone https://github.com/Gitcatmeoww/HITS-system-implementation.git
cd HITS-system-implementation
```

#### Setting Up a Virtual Environment (Optional)

- On macOS and Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

- On Windows:

```bash
py -m venv venv
.\venv\Scripts\activate
```

#### Backend Setup

- Install the required packages:

```bash
pip install -r requirements.txt
```

- Set up the environment variables according to `.env.example`:

  - This involves creating a .env file and populating it with your configuration details

- Initialize the database:

```bash
python backend/app/db/construct_db.py
```

- Start the Flask backend server:

```bash
python backend/app/app.py
```

#### Frontend Setup

- Navigate to the frontend application directory:

```bash
cd frontend/my-app
```

- Install the necessary npm packages:

```bash
npm install
```

- Start the frontend React application:

```bash
npm start
```

- The application should now be running at http://localhost:3000 and is ready for use

### Troubleshooting

This section addresses common issues you may encounter while setting up the application. Follow these steps to troubleshoot and resolve the issues.

- **`ModuleNotFoundError`: No module named 'backend.app'**

This error indicates that the application cannot locate the required module, possibly due to environment variables not being set correctly. Perform the following steps:

```bash
chmod +x setenv.sh  # Ensure the setenv.sh script is executable
source ./setenv.sh  # Execute the setenv.sh script to set the necessary environment variables
```

- **Database connection issue:**

If you're experiencing difficulties connecting to the PostgreSQL database, follow these steps for a possible solution:

- Connect to PostgreSQL server as a superuser, often the default `postgres` user:

```bash
psql -U postgres
```

- Create the database:

```bash
CREATE DATABASE dbname;
```

## Usage

After launching the frontend application, you will be presented with an interface that allows you to interact with the chatbot, explore datasets, and view detailed metadata for each dataset (under development)
