# llm-frameworks

Python toolkit for working with Large Language Models (LLMs) - modular wrappers, adapters, pipelines for text completion, chat, summarization, with CLI and Flask web interface

## Features

- **Modular Architecture**: Support for multiple LLM providers through adapter pattern
- **Multiple Providers**: OpenAI, HuggingFace, and extensible for more
- **Flexible APIs**: Text completion, chat, summarization capabilities
- **CLI & Web Interface**: Command-line tools and Flask-based web interface
- **Easy Configuration**: Environment-based configuration for API keys

## Installation

```bash
# Clone the repository
git clone https://github.com/rahit91890/llm-frameworks.git
cd llm-frameworks

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### 1. Environment Setup

Copy the `.env.example` file to create your own `.env` file:

```bash
cp .env.example .env
```

### 2. API Keys and Environment Variables

Edit the `.env` file with your API credentials:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo  # or gpt-4, gpt-4-turbo, etc.

# HuggingFace Configuration
HF_TOKEN=your_huggingface_token_here
HF_MODEL=gpt2  # or any other HuggingFace model

# Provider Selection
DEFAULT_PROVIDER=openai  # Options: openai, huggingface

# Flask Configuration (Optional)
FLASK_ENV=development
FLASK_PORT=5000
```

### 3. Getting API Keys

#### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

#### HuggingFace Token
1. Go to [HuggingFace](https://huggingface.co/)
2. Sign up or log in to your account
3. Go to Settings → Access Tokens
4. Create a new token
5. Copy the token to your `.env` file

## Usage

### Provider Selection

The framework supports multiple LLM providers. You can select your provider in two ways:

1. **Environment Variable** (Default): Set `DEFAULT_PROVIDER` in your `.env` file
2. **Runtime Selection**: Specify the provider when initializing adapters

### Supported Providers

- **OpenAI**: GPT-3.5, GPT-4, and other OpenAI models
- **HuggingFace**: Access to thousands of open-source models

### Basic Usage Example

```python
from adapters.openai_adapter import OpenAIAdapter
from adapters.huggingface_adapter import HuggingFaceAdapter

# Using OpenAI
openai_adapter = OpenAIAdapter()
response = openai_adapter.complete("Hello, how are you?")
print(response)

# Using HuggingFace
hf_adapter = HuggingFaceAdapter()
response = hf_adapter.complete("Hello, how are you?")
print(response)
```

## Architecture

```
llm-frameworks/
├── adapters/              # LLM provider adapters
│   ├── base_adapter.py    # Base adapter interface
│   ├── openai_adapter.py  # OpenAI implementation
│   └── huggingface_adapter.py  # HuggingFace implementation
├── llm.py                 # Main LLM framework
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
└── README.md             # This file
```

## Requirements

- Python 3.8+
- OpenAI API key (for OpenAI provider)
- HuggingFace token (for HuggingFace provider)
- See `requirements.txt` for package dependencies

## Security Notes

- **Never commit your `.env` file** - It contains sensitive API keys
- The `.gitignore` file is configured to exclude `.env` files
- Keep your API keys secure and rotate them regularly
- Monitor your API usage to avoid unexpected charges

## Contributing

Contributions are welcome! Feel free to:
- Add new provider adapters
- Improve existing functionality
- Report bugs or suggest features

## License

MIT License - see LICENSE file for details

## Support

For issues, questions, or contributions, please open an issue on GitHub.
