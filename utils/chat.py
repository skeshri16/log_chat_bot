from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import time
import json
from datetime import datetime
import re

# Import Azure OpenAI functionality
try:
    from .azure_openai import get_azure_chat_chain, run_azure_chat, get_azure_models, AzureOpenAIChatChain
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("Azure OpenAI not available. Install openai and langchain-openai packages.")

class RateLimitError(Exception):
    """Custom exception for rate limit errors with wait time information."""
    def __init__(self, message, wait_time=None, model_name=None, usage_info=None):
        self.message = message
        self.wait_time = wait_time
        self.model_name = model_name
        self.usage_info = usage_info or {}
        super().__init__(self.message)

def parse_rate_limit_error(error_message):
    """Extract detailed information from rate limit error message."""
    # Look for patterns like "Please try again in 6.54s"
    retry_pattern = r"Please try again in (\d+\.?\d*)s"
    wait_match = re.search(retry_pattern, str(error_message))
    wait_time = float(wait_match.group(1)) if wait_match else None
    
    # Extract model name
    model_pattern = r"model (\S+)"
    model_match = re.search(model_pattern, str(error_message))
    model_name = model_match.group(1) if model_match else "unknown"
    
    # Extract usage information
    usage_info = {}
    used_pattern = r"Used (\d+)"
    limit_pattern = r"Limit (\d+)"
    requested_pattern = r"Requested (\d+)"
    
    used_match = re.search(used_pattern, str(error_message))
    limit_match = re.search(limit_pattern, str(error_message))
    requested_match = re.search(requested_pattern, str(error_message))
    
    if used_match:
        usage_info['used'] = int(used_match.group(1))
    if limit_match:
        usage_info['limit'] = int(limit_match.group(1))
    if requested_match:
        usage_info['requested'] = int(requested_match.group(1))
    
    return wait_time, model_name, usage_info

def create_rate_limit_message(wait_time, model_name, usage_info):
    """Create a user-friendly rate limit message."""
    message = f"⏱️ Rate limit reached for {model_name}."
    
    if usage_info.get('used') and usage_info.get('limit'):
        message += f" Used {usage_info['used']}/{usage_info['limit']} tokens per minute."
    
    if wait_time:
        message += f" Please wait {wait_time:.1f} seconds before trying again."
    else:
        message += " Please wait a moment before trying again."
    
    message += " 💡 Tip: Try using a smaller model like 'Llama 3.1 8B Instant' to avoid rate limits."
    
    return message
def calculate_intelligent_delay(usage_info, base_delay=10):
    """Calculate delay based on current usage to avoid rate limits."""
    if not usage_info:
        return base_delay
    
    used = usage_info.get('used', 0)
    limit = usage_info.get('limit', 100000)
    
    if used == 0 or limit == 0:
        return base_delay
    
    # Calculate usage percentage
    usage_percent = used / limit
    
    if usage_percent > 0.95:  # >95% usage
        return 60  # Wait 1 minute
    elif usage_percent > 0.90:  # >90% usage  
        return 30  # Wait 30 seconds
    elif usage_percent > 0.80:  # >80% usage
        return 15  # Wait 15 seconds
    else:
        return base_delay

def run_model_comparison_smart(chat_states, query, test_name=""):
    """Simple model comparison since all models are now rate-limit friendly."""
    results = []
    
    print("🟢 Testing all available models...")
    for i, (model_name, chat_state) in enumerate(chat_states.items()):
        if i > 0:
            print(f"Waiting 1s before testing {model_name}...")
            time.sleep(1)
        
        try:
            print(f"Testing {model_name}...")
            response, _, metrics = run_chat(chat_state, query)
            results.append({
                "model_name": model_name,
                "response": response,
                "metrics": metrics,
                "success": True,
                "test_name": test_name,
                "error_type": "none"
            })
            print(f"✅ {model_name} completed in {metrics['total_time']}s")
            
        except RateLimitError as e:
            print(f"⚠️ {model_name} hit rate limit unexpectedly")
            response_message = getattr(e, 'message', None) or str(e) or "Rate limit exceeded"
            
            # Extract detailed rate limit information
            wait_time = getattr(e, 'wait_time', None)
            usage_info = getattr(e, 'usage_info', {})
            
            # Create enhanced metrics for rate limit
            rate_limit_metrics = {
                "wait_time": wait_time,
                "usage_info": usage_info,
                "total_time": 0,  # No processing time due to rate limit
                "llm_time": 0,
                "response_length": len(response_message) if response_message else 0,
                "error_details": {
                    "type": "rate_limit",
                    "model": model_name,
                    "timestamp": time.time()
                }
            }
            
            results.append({
                "model_name": model_name,
                "response": response_message,
                "metrics": rate_limit_metrics,
                "success": False,
                "test_name": test_name,
                "error_type": "rate_limit"
            })
        except Exception as e:
            print(f"❌ {model_name} error: {str(e)}")
            error_message = str(e) if e else "Unknown error occurred"
            results.append({
                "model_name": model_name,
                "response": f"Error: {error_message}",
                "metrics": None,
                "success": False,
                "test_name": test_name,
                "error_type": "other"
            })
    
    return results

# Available models for testing (Only rate-limit friendly models)
AVAILABLE_MODELS = {
    "Llama 3.1 8B Instant": "llama-3.1-8b-instant",  # ✅ Works well
    "OpenAI GPT-OSS 120B": "openai/gpt-oss-120b",    # ✅ Works well  
    "OpenAI GPT-OSS 20B": "openai/gpt-oss-20b",      # ✅ Works well
}

# Azure OpenAI models (if available)
AZURE_MODELS = {}
if AZURE_AVAILABLE:
    AZURE_MODELS = {
        "Azure GPT-4o Mini": "gpt-4o-mini",
        "Azure GPT-4o": "gpt-4o"
        # Note: GPT-4 and GPT-3.5 Turbo require additional access permissions
    }

def get_available_models():
    """Return list of available models for testing."""
    models = list(AVAILABLE_MODELS.keys())
    if AZURE_AVAILABLE:
        models.extend(list(AZURE_MODELS.keys()))
    return models

def get_all_models():
    """Return all models (same as available since we removed rate-limited ones)."""
    all_models = {**AVAILABLE_MODELS}
    if AZURE_AVAILABLE:
        all_models.update(AZURE_MODELS)
    return list(all_models.keys())

def get_model_id(model_name):
    """Get model ID from friendly name."""
    all_models = {**AVAILABLE_MODELS}
    if AZURE_AVAILABLE:
        all_models.update(AZURE_MODELS)
    return all_models.get(model_name, "llama-3.1-8b-instant")

def get_chat_chain(vectorstore, groq_api_key, model_name="Llama 3.1 8B Instant"):
    """Custom chain with retrieval + chat history injection."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Check if it's an Azure model
    if model_name in AZURE_MODELS:
        # For Azure models, we'll handle them differently
        return {
            "type": "azure",
            "retriever": retriever,
            "model_name": model_name,
            "model_id": get_model_id(model_name)
        }

    # Get the actual model ID from the friendly name for Groq models
    model_id = get_model_id(model_name)
    
    llm = ChatGroq(
        model_name=model_id,
        api_key=groq_api_key,
        temperature=0
    )

    template = """
You are a helpful assistant for analyzing log files.
Use the provided context chunks to answer queries.
If answer is not in the logs, say "I couldn't find this in the logs."

Chat History:
{chat_history}

Context:
{context}

User Query:
{query}

Answer:
"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "query"],
        template=template,
    )

    return {
        "type": "groq",
        "llm": llm, 
        "prompt": prompt, 
        "retriever": retriever, 
        "model_name": model_name,
        "model_id": model_id
    }

def get_azure_chat_chain_configured(vectorstore, client_id, client_secret, model_name="Azure GPT-4o Mini"):
    """Get configured Azure chat chain for the chatbot."""
    if not AZURE_AVAILABLE:
        raise Exception("Azure OpenAI not available. Please install required packages.")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    deployment_name = get_model_id(model_name)
    
    azure_chain = get_azure_chat_chain(retriever, client_id, client_secret, deployment_name)
    
    return {
        "type": "azure",
        "azure_chain": azure_chain,
        "retriever": retriever,
        "model_name": model_name,
        "model_id": deployment_name
    }

def run_chat(chat_state, query):
    """Run retrieval + inject chat history into prompt with performance tracking and rate limit handling."""
    start_time = time.time()
    
    # Retrieval phase
    retrieval_start = time.time()
    retrieved_docs = chat_state["retriever"].get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    retrieval_time = time.time() - retrieval_start

    # Handle Azure models differently
    if chat_state.get("type") == "azure":
        try:
            llm_start = time.time()
            azure_result = run_azure_chat(chat_state["azure_chain"], chat_state["retriever"], query)
            llm_time = time.time() - llm_start
            
            if azure_result["status"] == "success":
                response = azure_result["response"]
                success = True
                error_message = None
            else:
                response = azure_result["response"]
                success = False
                error_message = azure_result["response"]
                
        except Exception as e:
            llm_time = time.time() - llm_start
            success = False
            error_message = f"Error with {chat_state.get('model_name', 'Azure model')}: {str(e)}"
            response = f"❌ {error_message}"
    else:
        # Handle Groq models (original logic)
        # prepare prompt
        prompt_text = chat_state["prompt"].format(
            chat_history="\n".join(chat_state.get("history", [])),
            context=context,
            query=query
        )

        # LLM inference phase with rate limit handling
        try:
            llm_start = time.time()
            response = chat_state["llm"].predict(prompt_text)
            llm_time = time.time() - llm_start
            success = True
            error_message = None
            
        except Exception as e:
            llm_time = time.time() - llm_start
            error_str = str(e)
            
            # Check if it's a rate limit error
            if "rate_limit_exceeded" in error_str or "Rate limit reached" in error_str:
                wait_time, model_name, usage_info = parse_rate_limit_error(error_str)
                user_message = create_rate_limit_message(
                    wait_time, 
                    chat_state.get('model_name', model_name), 
                    usage_info
                )
                
                success = False
                error_message = user_message
                response = user_message
                
                # Raise custom rate limit error with details
                raise RateLimitError(
                    user_message, 
                    wait_time=wait_time, 
                    model_name=chat_state.get('model_name', model_name),
                    usage_info=usage_info
                )
            else:
                # Handle other errors
                success = False
                error_message = f"Error with {chat_state.get('model_name', 'model')}: {str(e)}"
                response = f"❌ {error_message}"
    
    total_time = time.time() - start_time

    # update history only if successful
    if success:
        chat_state.setdefault("history", []).append(f"User: {query}")
        chat_state["history"].append(f"Assistant: {response}")

    # Store performance metrics
    performance_metrics = {
        "model_name": chat_state.get("model_name", "Unknown"),
        "model_id": chat_state.get("model_id", "Unknown"),
        "total_time": round(total_time, 3),
        "retrieval_time": round(retrieval_time, 3), 
        "llm_time": round(llm_time, 3),
        "context_chunks": len(retrieved_docs),
        "context_length": len(context),
        "response_length": len(response),
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "error_message": error_message
    }
    
    chat_state.setdefault("performance_history", []).append(performance_metrics)

    return response, chat_state["history"], performance_metrics

def run_model_comparison(chat_states, query, test_name=""):
    """Run the same query across multiple models for comparison with rate limit handling."""
    results = []
    
    for model_name, chat_state in chat_states.items():
        try:
            start_time = time.time()
            response, _, metrics = run_chat(chat_state, query)
            
            result = {
                "model_name": model_name,
                "response": response,
                "metrics": metrics,
                "success": True,
                "test_name": test_name,
                "error_type": "none"
            }
            results.append(result)
            
        except RateLimitError as e:
            response_message = getattr(e, 'message', None) or str(e) or "Rate limit exceeded"
            result = {
                "model_name": model_name,
                "response": response_message,
                "metrics": {
                    "total_time": 0,
                    "llm_time": 0,
                    "retrieval_time": 0,
                    "wait_time": getattr(e, 'wait_time', None),
                    "usage_info": getattr(e, 'usage_info', {})
                },
                "success": False,
                "test_name": test_name,
                "error_type": "rate_limit",
                "wait_time": getattr(e, 'wait_time', None)
            }
            results.append(result)
            
        except Exception as e:
            error_message = str(e) if e else "Unknown error occurred"
            result = {
                "model_name": model_name,
                "response": f"Error: {error_message}",
                "metrics": None,
                "success": False,
                "test_name": test_name,
                "error_type": "other"
            }
            results.append(result)
    
    return results

def run_model_comparison_with_staggered_requests(chat_states, query, test_name="", delay_between_requests=2):
    """
    Run model comparison with delays between requests to avoid hitting rate limits simultaneously.
    """
    results = []
    
    for i, (model_name, chat_state) in enumerate(chat_states.items()):
        if i > 0:  # Add delay between requests (except for the first one)
            print(f"Waiting {delay_between_requests}s before testing {model_name}...")
            time.sleep(delay_between_requests)
        
        try:
            print(f"Testing {model_name}...")
            start_time = time.time()
            response, _, metrics = run_chat(chat_state, query)
            
            result = {
                "model_name": model_name,
                "response": response,
                "metrics": metrics,
                "success": metrics.get("success", True),
                "test_name": test_name,
                "error_type": "none" if metrics.get("success", True) else "rate_limit" if "rate limit" in metrics.get("error_message", "").lower() else "other"
            }
            results.append(result)
            
            if not metrics.get("success", True):
                print(f"⚠️ {model_name} failed: {metrics.get('error_message', 'Unknown error')}")
            else:
                print(f"✅ {model_name} completed in {metrics['total_time']}s")
            
        except Exception as e:
            print(f"❌ {model_name} exception: {str(e)}")
            result = {
                "model_name": model_name,
                "response": f"Error: {str(e)}",
                "metrics": None,
                "success": False,
                "test_name": test_name,
                "error_type": "exception"
            }
            results.append(result)
    
    return results

def get_performance_summary(chat_state):
    """Get performance summary for a model."""
    if "performance_history" not in chat_state:
        return None
    
    history = chat_state["performance_history"]
    if not history:
        return None
    
    total_queries = len(history)
    avg_total_time = sum(m["total_time"] for m in history) / total_queries
    avg_llm_time = sum(m["llm_time"] for m in history) / total_queries
    avg_retrieval_time = sum(m["retrieval_time"] for m in history) / total_queries
    
    return {
        "model_name": history[0]["model_name"],
        "total_queries": total_queries,
        "avg_total_time": round(avg_total_time, 3),
        "avg_llm_time": round(avg_llm_time, 3),
        "avg_retrieval_time": round(avg_retrieval_time, 3),
        "min_total_time": round(min(m["total_time"] for m in history), 3),
        "max_total_time": round(max(m["total_time"] for m in history), 3)
    }
