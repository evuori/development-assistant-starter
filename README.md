# Development Assistant Starter

A Streamlit-based application that leverages AI to assist with software development tasks. The application uses LangChain, OpenAI, and LangGraph to create an intelligent coding assistant that can help with code generation, testing, and debugging.

## Features

- Interactive web interface built with Streamlit
- AI-powered code generation and refinement
- Automated test case generation
- Code execution and error handling
- Multi-agent workflow for comprehensive code development
- Environment setup and configuration management

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Azure OpenAI API credentials (if using Azure)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd software-development-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env-example` to `.env`
   - Fill in your API keys and configuration settings

## Project Structure

- `app.py`: Main application file with Streamlit interface
- `agents.py`: Agent definitions and setup
- `models.py`: Data models and schemas
- `workflow.py`: LangGraph workflow implementation
- `.streamlit/`: Streamlit configuration directory
- `.env`: Environment variables and API keys
- `requirements.txt`: Project dependencies

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Use the interface to:
   - Input your coding requirements
   - Generate and refine code
   - Run tests
   - Debug and fix issues

## Dependencies

- streamlit >= 1.32.0
- langchain >= 0.1.0
- langchain-openai >= 0.0.5
- langgraph >= 0.0.15
- langchain-core >= 0.1.0
- pydantic >= 2.0.0
- duckduckgo-search >= 4.1.1
- python-dotenv >= 1.0.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here]

## Acknowledgments

- OpenAI for providing the language models
- LangChain team for the framework
- Streamlit team for the web interface framework 