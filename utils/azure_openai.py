"""
Azure OpenAI integration for the chatbot system.
Uses client credentials (client_id and client_secret) for authentication.
"""

import requests
import base64
import json
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import time
from typing import Dict, List, Optional


class AzureOpenAIClient:
    """Azure OpenAI client with client credentials authentication."""
    
    def __init__(self, client_id: str, client_secret: str, 
                 endpoint: str = "https://chat-ai.cisco.com",
                 deployment_name: str = "gpt-4o-mini",
                 api_version: str = "2024-07-01-preview"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.oauth_url = "https://id.cisco.com/oauth2/default/v1/token"
        self.app_key = "egai-prd-mig-routing-ai-ci-cd-xr-github-pipeline-1"
        
        # Initialize client
        self.access_token = None
        self.client = None
        self._refresh_token()
    
    def _get_azure_token(self) -> str:
        """Get OAuth token using client credentials."""
        payload = "grant_type=client_credentials"
        value = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode("utf-8")).decode("utf-8")
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {value}"
        }

        response = requests.post(self.oauth_url, headers=headers, data=payload)
        response_data = response.json()
        
        if response.status_code != 200:
            raise Exception(f"Failed to get Azure token: {response_data}")
            
        return response_data.get("access_token", None)
    
    def _refresh_token(self):
        """Refresh the access token and reinitialize the client."""
        try:
            self.access_token = self._get_azure_token()
            if not self.access_token:
                raise Exception("Failed to obtain access token")
                
            self.client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.access_token,
                api_version=self.api_version
            )
        except Exception as e:
            print(f"Error refreshing Azure token: {e}")
            raise
    
    def chat_completion(self, messages: List[Dict], temperature: float = 0.7, 
                       max_tokens: int = 1000) -> str:
        """Get chat completion from Azure OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                user=json.dumps({"appkey": self.app_key})
            )
            
            if response and hasattr(response, "choices") and response.choices:
                return response.choices[0].message.content
            else:
                return "No response generated"
                
        except Exception as e:
            # Try refreshing token once if unauthorized
            if "401" in str(e) or "unauthorized" in str(e).lower():
                try:
                    self._refresh_token()
                    response = self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        user=json.dumps({"appkey": self.app_key})
                    )
                    
                    if response and hasattr(response, "choices") and response.choices:
                        return response.choices[0].message.content
                    else:
                        return "No response generated"
                except Exception as retry_error:
                    raise Exception(f"Azure OpenAI error (after token refresh): {retry_error}")
            else:
                raise Exception(f"Azure OpenAI error: {e}")


class AzureOpenAIChatChain:
    """LangChain-compatible Azure OpenAI chat chain."""
    
    def __init__(self, azure_client: AzureOpenAIClient, system_prompt: str = ""):
        self.azure_client = azure_client
        self.system_prompt = system_prompt
        self.conversation_history = []
    
    def run(self, human_input: str, context: str = "") -> str:
        """Run the chat chain with human input and optional context."""
        # Prepare messages
        messages = []
        
        # Add system prompt if provided
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add conversation history
        for msg in self.conversation_history[-5:]:  # Keep last 5 exchanges
            messages.append(msg)
        
        # Add current human input with context
        if context:
            user_content = f"Context: {context}\n\nQuestion: {human_input}"
        else:
            user_content = human_input
            
        messages.append({"role": "user", "content": user_content})
        
        # Get response
        response = self.azure_client.chat_completion(messages)
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": human_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response


def get_azure_chat_chain(retriever, client_id: str, client_secret: str, 
                        deployment_name: str = "gpt-4o-mini") -> AzureOpenAIChatChain:
    """Create Azure OpenAI chat chain for RAG."""
    
    system_prompt = """You are a helpful AI assistant that analyzes log files and system data.
    You provide detailed, accurate responses based on the provided context.
    If you cannot find relevant information in the context, say so clearly.
    Focus on being precise and helpful in your analysis."""
    
    azure_client = AzureOpenAIClient(
        client_id=client_id,
        client_secret=client_secret,
        deployment_name=deployment_name
    )
    
    return AzureOpenAIChatChain(azure_client, system_prompt)


def run_azure_chat(chat_chain, retriever, query: str):
    """Run Azure chat with retrieval context."""
    start_time = time.time()
    
    try:
        # Get relevant context from retriever
        relevant_docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in relevant_docs[:3]])
        
        # Get response from Azure OpenAI
        response = chat_chain.run(query, context)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            "response": response,
            "response_time": response_time,
            "context_docs": len(relevant_docs),
            "status": "success"
        }
        
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            "response": f"Error: {str(e)}",
            "response_time": response_time,
            "context_docs": 0,
            "status": "error"
        }


# Available Azure OpenAI models
AZURE_MODELS = {
    "Azure GPT-4o Mini": "gpt-4o-mini",
    "Azure GPT-4o": "gpt-4o"
    # Note: GPT-4 and GPT-3.5 Turbo require additional access permissions
}

def get_azure_models() -> List[str]:
    """Get list of available Azure OpenAI models."""
    return list(AZURE_MODELS.keys())

def get_azure_deployment_name(model_display_name: str) -> str:
    """Get deployment name for Azure model."""
    return AZURE_MODELS.get(model_display_name, "gpt-4o-mini")
