# 🔍 Log Chat Boat - Intelligent Log Analysis Chatbot

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](htt### 🔒 Security & Privacy

### API Security
- **Secure Storage**: API keys stored in session state only, never persistent
- **User-Provided Credentials**: All credentials entered by users, not hardcoded
- **Enterprise Integration**: Azure OpenAI with organization-provided credentials
- **No Data Persistence**: Logs processed in memory only
- **Encrypted Connections**: HTTPS for all API communicationsreamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-00D4AA.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An intelligent chatbot system for analyzing log files using multiple AI models with RAG (Retrieval-Augmented Generation) capabilities.**

## 🌟 Overview

Log Chat Boat is a sophisticated log analysis tool that combines the power of modern AI models with document retrieval to provide intelligent insights from your log files. It supports both Groq and Azure OpenAI models, offers performance comparison features, and includes comprehensive API testing capabilities.

### ✨ Key Features

- 🤖 **Multi-Model Support**: Groq (Llama models) + Azure OpenAI (GPT-4, GPT-4o)
- 📊 **Performance Comparison**: Compare response times and accuracy across models
- 🔍 **RAG Implementation**: Retrieval-Augmented Generation for accurate log analysis
- 🧪 **API Testing**: Built-in connection testing for all supported APIs
- 📈 **Performance Metrics**: Detailed analytics and visualization
- 🎨 **Interactive UI**: Clean Streamlit interface with real-time feedback
- 🛡️ **Rate Limit Management**: Smart handling of API rate limits
- 🔐 **Enterprise Security**: Azure OpenAI integration with client credentials

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Virtual environment (recommended)
- API keys (Groq API key and/or Azure OpenAI credentials)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/skeshri16/log_chat_boat.git
   cd log_chat_boat
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```
5. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Configure your API keys
   - Upload a log file and start chatting!

## 🔧 Configuration

### API Setup

#### Groq API
1. Get your API key from [Groq Console](https://console.groq.com)
2. Enter your API key in the "Groq API Key" field

#### Azure OpenAI (Optional)
1. Enter your Azure Client ID and Client Secret
2. These credentials are provided by your organization's Azure administrator

### Supported Models

#### 🟢 Groq Models (Rate-Limit Friendly)
- **Llama 3.1 8B Instant** - Fast and efficient
- **OpenAI GPT-OSS 120B** - Large context window
- **OpenAI GPT-OSS 20B** - Balanced performance

#### 🔵 Azure OpenAI Models (Enterprise)
- **Azure GPT-4o Mini** - Fast enterprise model
- **Azure GPT-4o** - Balanced enterprise performance

## 📋 Usage Modes

### 1. Single Model Chat
- Select one model for interactive conversation
- Real-time performance metrics
- Conversation history tracking
- Perfect for focused analysis

### 2. Model Comparison
- Compare multiple models side-by-side
- Performance benchmarking
- Response quality comparison
- Choose between Smart (rate-limit aware) and Standard modes

### 3. Performance Analysis
- Detailed metrics visualization
- Response time analysis
- Model efficiency comparison
- Historical performance tracking

## 🎯 Features Deep Dive

### 🧠 RAG (Retrieval-Augmented Generation)
- **Document Chunking**: Intelligent text segmentation
- **Vector Embeddings**: Sentence transformers for semantic search
- **FAISS Vector Store**: Fast similarity search
- **Context Retrieval**: Relevant document chunks for accurate responses

### 📊 Performance Metrics
- **Response Time**: Total, LLM, and retrieval time tracking
- **Context Analysis**: Document chunks and context length
- **Success Rate**: Error handling and success tracking
- **Comparative Analysis**: Side-by-side model performance

### 🛡️ Rate Limit Management
- **Smart Comparison**: Automatically skips rate-limited models
- **Intelligent Delays**: Dynamic wait times based on usage
- **Error Recovery**: Graceful handling of API limits
- **User Messaging**: Clear explanations and suggestions

### 🧪 API Testing & Diagnostics
- **Connection Testing**: Verify API keys before use
- **Real-time Status**: Visual indicators for API health
- **Error Diagnostics**: Detailed error messages and solutions
- **Bulk Testing**: Test all APIs simultaneously

## 📁 Project Structure

```
log_chat_boat/
├── app.py                      # Main Streamlit application
├── requirement.txt             # Python dependencies
├── utils/                      # Utility modules
│   ├── chat.py                # Chat logic and model management
│   ├── loader.py              # Document loading and chunking
│   ├── vectorstore.py         # Vector database operations
│   ├── azure_openai.py        # Azure OpenAI integration
│   └── api_test.py            # API testing utilities

```

## 🎨 User Interface

### Sidebar Configuration
- **API Selection**: Checkboxes to enable/disable Groq and Azure OpenAI
- **Model Selection**: Automatically filtered based on enabled APIs
- **Mode Selection**: Single chat, comparison, or analysis

### Main Interface
- **API Keys**: Secure input fields for credentials
- **Status Indicators**: Real-time API configuration status
- **File Upload**: Drag-and-drop log file upload
- **Chat Interface**: Natural language queries about logs
- **Results Display**: Formatted responses with metrics
- **Comparison View**: Side-by-side model comparison
- **Performance Charts**: Interactive visualizations

  
## 🛠️ Technical Architecture

### Core Technologies
- **Frontend**: Streamlit (Python web framework)
- **AI Framework**: LangChain for LLM orchestration
- **Vector Database**: FAISS for similarity search
- **Embeddings**: Sentence Transformers
- **APIs**: Groq API, Azure OpenAI API
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly for interactive charts

### Key Components

#### 1. Document Processing Pipeline
```python
Log File → Text Extraction → Chunking → Embedding → Vector Store
```

#### 2. RAG Query Flow
```python
User Query → Embedding → Similarity Search → Context Retrieval → LLM → Response
```

#### 3. Multi-Model Architecture
```python
Query → Model Router → [Groq Models | Azure Models] → Response Aggregation
```

## 📈 Performance Optimization

### Rate Limit Strategies
- **Smart Model Selection**: Prefer rate-limit friendly models
- **Staggered Requests**: Delays between API calls
- **Error Recovery**: Automatic retry with backoff
- **User Guidance**: Clear messaging about limitations

### Response Time Optimization
- **Efficient Chunking**: Optimized document segmentation
- **Context Limiting**: Balanced context window usage
- **Parallel Processing**: Concurrent model queries where possible
- **Caching**: Session state management for repeated queries

## 🔒 Security & Privacy

### API Security
- **Secure Storage**: API keys stored in session state only
- **Enterprise Integration**: Azure OpenAI with client credentials
- **No Data Persistence**: Logs processed in memory only
- **Encrypted Connections**: HTTPS for all API communications

### Data Handling
- **Local Processing**: Document processing happens locally
- **No Data Logging**: User data not stored or logged
- **Session Isolation**: Each session is independent
- **Memory Management**: Automatic cleanup of processed data

## 🧪 Quality Assurance

### Quality Features
- **Type Hints**: Comprehensive type annotations
- **Error Handling**: Graceful failure management
- **Logging**: Detailed operation logging
- **Monitoring**: Real-time performance tracking

## 📚 Documentation

### Available Guides
- **[API Testing Guide](API_TESTING_GUIDE.md)**: Complete testing features documentation
- **[Azure OpenAI Guide](AZURE_OPENAI_GUIDE.md)**: Enterprise integration setup
- **[Model Comparison Guide](MODEL_COMPARISON_GUIDE.md)**: Performance comparison features

### Code Documentation
- **Inline Comments**: Detailed code explanations
- **Function Docstrings**: Comprehensive API documentation
- **Type Annotations**: Clear parameter and return types
- **Architecture Diagrams**: Visual system overview

## 🚀 Advanced Usage

### Custom Model Configuration
```python
# Add custom models to chat.py
CUSTOM_MODELS = {
    "Custom Model Name": "model-id",
    # Add your models here
}
```

### Performance Tuning
```python
# Adjust chunking parameters in loader.py
chunk_size = 1000  # Increase for larger context
chunk_overlap = 200  # Adjust overlap for better retrieval
```

### API Integration
```python
# Extend with new APIs in utils/
from your_api import YourAPIClient
# Follow existing patterns for integration
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

### Contribution Guidelines
- **Code Style**: Follow PEP 8 standards
- **Testing**: Add tests for new features
- **Documentation**: Update relevant docs
- **Type Hints**: Use comprehensive type annotations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎉 Acknowledgments

- **LangChain**: For the excellent LLM framework
- **Streamlit**: For the intuitive web framework
- **Groq**: For fast LLM inference
- **Azure OpenAI**: For enterprise-grade AI models
- **FAISS**: For efficient vector similarity search

## 📞 Support

### Getting Help
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join GitHub Discussions for questions
- **Documentation**: Check the docs/ directory for guides

### Common Issues
- **Rate Limits**: Use Azure models or wait between requests
- **API Keys**: Verify keys in respective consoles
- **Performance**: Try smaller models for faster responses
- **Memory**: Use smaller chunk sizes for large files

---

**Ready to analyze your logs intelligently? Start with `streamlit run app.py`!** 🚀
