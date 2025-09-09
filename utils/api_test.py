"""
API connection testing utilities for both Groq and Azure OpenAI.
"""

def test_groq_connection(api_key):
    """Test Groq API connection."""
    try:
        from langchain_groq import ChatGroq
        
        test_llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            api_key=api_key,
            temperature=0
        )
        
        test_response = test_llm.predict("Say 'Connection test successful'")
        
        return {
            "success": True,
            "message": "Groq API connection successful",
            "response": test_response[:100] + "..." if len(test_response) > 100 else test_response
        }
        
    except Exception as e:
        error_msg = str(e).lower()
        
        if "invalid api key" in error_msg or "authentication" in error_msg:
            return {
                "success": False,
                "message": "Invalid Groq API key",
                "error": "Please check your API key"
            }
        elif "rate_limit" in error_msg:
            return {
                "success": False,
                "message": "Rate limit reached",
                "error": "Please wait a moment and try again"
            }
        else:
            return {
                "success": False,
                "message": "Connection failed",
                "error": str(e)
            }

def test_azure_connection(client_id, client_secret):
    """Test Azure OpenAI connection."""
    try:
        from .azure_openai import AzureOpenAIClient
        
        azure_client = AzureOpenAIClient(client_id, client_secret)
        
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Connection test successful'"}
        ]
        
        test_response = azure_client.chat_completion(test_messages, max_tokens=50)
        
        return {
            "success": True,
            "message": "Azure OpenAI connection successful",
            "response": test_response[:100] + "..." if len(test_response) > 100 else test_response,
            "endpoint": azure_client.endpoint,
            "deployment": azure_client.deployment_name
        }
        
    except Exception as e:
        error_msg = str(e).lower()
        
        if "401" in error_msg or "unauthorized" in error_msg:
            return {
                "success": False,
                "message": "Authentication failed",
                "error": "Please check your Client ID and Secret"
            }
        elif "token" in error_msg:
            return {
                "success": False,
                "message": "Token error",
                "error": "Unable to obtain access token"
            }
        else:
            return {
                "success": False,
                "message": "Connection failed",
                "error": str(e)
            }

def get_available_models_with_status(groq_key=None, azure_enabled=False):
    """Get available models with their connection status."""
    from .chat import get_available_models, AZURE_AVAILABLE, AZURE_MODELS
    
    models_status = {}
    all_models = get_available_models()
    
    for model in all_models:
        if "Azure" in model:
            if AZURE_AVAILABLE and azure_enabled:
                models_status[model] = {"available": True, "type": "azure", "status": "Ready"}
            else:
                models_status[model] = {"available": False, "type": "azure", "status": "Disabled or not configured"}
        else:
            if groq_key:
                models_status[model] = {"available": True, "type": "groq", "status": "Ready"}
            else:
                models_status[model] = {"available": False, "type": "groq", "status": "API key missing"}
    
    return models_status

def get_connection_summary(groq_key=None, azure_enabled=False, azure_client_id=None, azure_client_secret=None):
    """Get a summary of all API connections."""
    from .chat import AZURE_AVAILABLE
    
    summary = {
        "groq": {
            "configured": bool(groq_key),
            "status": "Ready" if groq_key else "API key missing"
        },
        "azure": {
            "available": AZURE_AVAILABLE,
            "enabled": azure_enabled,
            "configured": bool(azure_client_id and azure_client_secret),
            "status": "Not available" if not AZURE_AVAILABLE else 
                     "Disabled" if not azure_enabled else
                     "Ready" if (azure_client_id and azure_client_secret) else
                     "Credentials missing"
        }
    }
    
    return summary
