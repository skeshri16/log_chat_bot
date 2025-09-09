import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.loader import load_and_chunk
from utils.vectorstore import build_vectorstore
from utils.chat import get_chat_chain, run_chat, run_model_comparison_smart, get_performance_summary, get_available_models, get_all_models, RateLimitError, get_azure_chat_chain_configured, AZURE_AVAILABLE, AZURE_MODELS
from groq import AuthenticationError
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.loader import load_and_chunk
from utils.vectorstore import build_vectorstore
from utils.chat import get_chat_chain, run_chat, run_model_comparison, run_model_comparison_with_staggered_requests, get_performance_summary, get_available_models
from groq import AuthenticationError
import json

st.set_page_config(page_title="Log Chatbot - Model Comparison", layout="wide")
st.title("🔍 Log File Chatbot with Model Performance Testing")

# Collapsible API Configuration in main area
with st.expander("🔑 API Configuration", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        # Groq API Configuration
        st.subheader("🟢 Groq API")
        use_groq = st.checkbox("Enable Groq Models", value=True, help="Enable to use Groq models for inference")
        if use_groq:
            groq_api_key = st.text_input(
                "Groq API Key:", 
                type="password", 
                placeholder="Enter your Groq API key here...",
                help="Get your API key from https://console.groq.com/"
            )
        else:
            groq_api_key = ""
    
    with col2:
        # Azure OpenAI Configuration
        if AZURE_AVAILABLE:
            st.subheader("🔵 Azure OpenAI")
            use_azure = st.checkbox("Enable Azure OpenAI Models", value=True, help="Enable to use Azure OpenAI models")
            if use_azure:
                azure_client_id = st.text_input(
                    "Azure Client ID:", 
                    type="password",
                    placeholder="Enter your Azure Client ID...",
                    help="Your Azure application client ID"
                )
                azure_client_secret = st.text_input(
                    "Azure Client Secret:", 
                    type="password",
                    placeholder="Enter your Azure Client Secret...",
                    help="Your Azure application client secret"
                )
            else:
                azure_client_id = ""
                azure_client_secret = ""
        else:
            st.info("🔵 Azure OpenAI not available. Install packages: `pip install openai langchain-openai`")
            use_azure = False
            azure_client_id = ""
            azure_client_secret = ""
    
    # Status indicators row
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if use_groq and groq_api_key:
            st.success("🟢 Groq Ready")
        elif use_groq and not groq_api_key:
            st.error("❌ Groq Key Missing")
        else:
            st.info("ℹ️ Groq Disabled")
    
    with col2:
        if AZURE_AVAILABLE and use_azure and azure_client_id and azure_client_secret:
            st.success("🔵 Azure Ready")
        elif not AZURE_AVAILABLE:
            st.warning("⚠️ Azure N/A")
        elif not use_azure:
            st.info("ℹ️ Azure Disabled") 
        elif use_azure and (not azure_client_id or not azure_client_secret):
            st.error("❌ Azure Incomplete")
        else:
            st.error("❌ Azure Error")
    
    # Quick setup tips
    st.markdown("**💡 Quick Setup Tips:**")
    st.markdown("- ✅ Configure at least one API to get started")
    st.markdown("- 🔒 Your credentials are stored securely in this session only")

# Check if at least one API is configured
api_configured = (use_groq and groq_api_key) or (use_azure and azure_client_id and azure_client_secret)

# File Upload Section
st.header("📁 Upload Log File")

if not api_configured:
    st.warning("⚠️ **Configuration Required:** Please configure at least one API (Groq or Azure) above before uploading files.")
    uploaded_file = None
else:
    uploaded_file = st.file_uploader(
        "Choose a .txt log file", 
        type=["txt"],
        help="Upload your log file to start analysis. Supported format: .txt"
    )

# Sidebar for testing mode selection
with st.sidebar:
    st.header("Testing Mode")
    
    # Add model legend
    st.info("""
    **Model Types:**
    
    🔵 Azure OpenAI Model
    
    🟢 Groq Model
    """)
    
    testing_mode = st.selectbox(
        "Select Mode:",
        ["Single Model Chat", "Model Comparison", "Performance Analysis"]
    )
    
    if testing_mode == "Single Model Chat":
        # Get available models and filter based on enabled APIs
        all_available_models = get_available_models()
        available_models = []
        
        for model in all_available_models:
            # Include Azure models only if Azure is enabled
            if "Azure" in model and use_azure:
                available_models.append(model)
            # Include Groq models only if Groq is enabled
            elif "Azure" not in model and use_groq:
                available_models.append(model)
        
        if not available_models:
            st.warning("⚠️ No models available. Please enable at least one API in the configuration.")
            st.stop()
        
        # Add visual indicators
        model_options = []
        for model in available_models:
            if "Azure" in model:
                model_options.append(f"🔵 {model}")
            else:
                model_options.append(f"🟢 {model}")
        
        selected_display = st.selectbox(
            "Select Model:",
            model_options
        )
        # Extract the actual model name (remove emoji prefix)
        selected_model = selected_display.split(" ", 1)[1] if " " in selected_display else selected_display
        
    elif testing_mode == "Model Comparison":
        st.write("Compare multiple models:")
        all_available_models = get_available_models()
        all_models = get_all_models()
        
        # Filter models based on enabled APIs
        available_models = []
        filtered_all_models = []
        
        for model in all_models:
            # Include Azure models only if Azure is enabled
            if "Azure" in model and use_azure:
                filtered_all_models.append(model)
                if model in all_available_models:
                    available_models.append(model)
            # Include Groq models only if Groq is enabled
            elif "Azure" not in model and use_groq:
                filtered_all_models.append(model)
                if model in all_available_models:
                    available_models.append(model)
        
        if not filtered_all_models:
            st.warning("⚠️ No models available. Please enable at least one API in the configuration.")
            st.stop()
        
        # Add visual indicators for all models (only Azure and Groq, no rate-limited)
        model_options = []
        for model in filtered_all_models:
            if "Azure" in model:
                model_options.append(f"🔵 {model}")
            else:
                model_options.append(f"🟢 {model}")
        
        # Create mapping from display name to actual name
        model_mapping = {}
        for i, model in enumerate(filtered_all_models):
            model_mapping[model_options[i]] = model
        
        selected_display_models = st.multiselect(
            "Select Models to Compare:",
            model_options,
            default=[opt for opt in model_options if any(avail in opt for avail in available_models)]
        )
        
        # Convert back to actual model names
        selected_models = [model_mapping[display] for display in selected_display_models]

# Initialize session state
if "chat_states" not in st.session_state:
    st.session_state.chat_states = {}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = []

if uploaded_file and api_configured:
    try:
        with st.spinner("📑 Chunking log file..."):
            chunks = load_and_chunk(uploaded_file)
        st.success(f"✅ Chunking done! Created {len(chunks)} chunks.")

        with st.spinner("🔍 Creating embeddings & building vectorstore..."):
            vectorstore = build_vectorstore(chunks)
            st.session_state.vectorstore = vectorstore
        st.success("✅ Embedding process completed!")

        # Initialize chat states based on mode
        if testing_mode == "Single Model Chat":
            if selected_model not in st.session_state.chat_states:
                # Check if it's an Azure model
                if use_azure and selected_model in AZURE_MODELS:
                    if not azure_client_id or not azure_client_secret:
                        st.error("❌ Please provide Azure Client ID and Client Secret for Azure models.")
                        st.stop()
                    st.session_state.chat_states[selected_model] = get_azure_chat_chain_configured(
                        vectorstore, azure_client_id, azure_client_secret, selected_model
                    )
                # Check if it's a Groq model and Groq is enabled
                elif use_groq and groq_api_key:
                    st.session_state.chat_states[selected_model] = get_chat_chain(
                        vectorstore, groq_api_key, selected_model
                    )
                else:
                    st.error("❌ Please configure the appropriate API for the selected model.")
                    st.stop()
            st.success(f"🚀 {selected_model} ready for chat!")
            
        elif testing_mode == "Model Comparison":
            with st.spinner("⚙️ Initializing selected models for comparison..."):
                for model in selected_models:
                    if model not in st.session_state.chat_states:
                        # Check if it's an Azure model
                        if use_azure and model in AZURE_MODELS:
                            if not azure_client_id or not azure_client_secret:
                                st.error("❌ Please provide Azure Client ID and Client Secret for Azure models.")
                                st.stop()
                            st.session_state.chat_states[model] = get_azure_chat_chain_configured(
                                vectorstore, azure_client_id, azure_client_secret, model
                            )
                        # Check if it's a Groq model and Groq is enabled
                        elif use_groq and groq_api_key:
                            st.session_state.chat_states[model] = get_chat_chain(
                                vectorstore, groq_api_key, model
                            )
                        else:
                            st.warning(f"⚠️ Skipping {model} - API not configured")
                            continue
            st.success(f"🚀 {len([m for m in selected_models if m in st.session_state.chat_states])} models ready for comparison!")
            st.info("💡 **Models are initialized but not running yet.** Enter a query below and click 'Run Comparison' to start the actual comparison.")

    except AuthenticationError:
        st.error("❌ API key invalid or expired. Please check your API keys.")
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
elif uploaded_file and not api_configured:
    st.warning("⚠️ Please configure at least one API (Groq or Azure OpenAI) to process the file.")
    st.info("💡 Enable Groq or Azure OpenAI in the sidebar and provide the required credentials.")

# Main interface based on mode
if st.session_state.vectorstore and api_configured:
    
    if testing_mode == "Single Model Chat":
        st.header(f"Chat with {selected_model}")
        
        query = st.text_input("Ask a question about the logs:")
        if query and selected_model in st.session_state.chat_states:
            try:
                with st.spinner("Generating response..."):
                    response, history, metrics = run_chat(
                        st.session_state.chat_states[selected_model], query
                    )
                
                # Display response and metrics
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Answer:** {response}")
                
                with col2:
                    st.metric("Response Time", f"{metrics['total_time']}s")
                    st.metric("LLM Time", f"{metrics['llm_time']}s")
                    st.metric("Retrieval Time", f"{metrics['retrieval_time']}s")
                    
            except RateLimitError as e:
                st.warning(e.message)
                if e.wait_time:
                    st.info(f"⏰ Please wait {e.wait_time:.1f} seconds before trying again.")
                st.info("💡 **Tip:** Try using 'Llama 3.1 8B Instant' for faster responses with fewer rate limits.")
            except AuthenticationError:
                st.error("❌ API key invalid or expired.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    elif testing_mode == "Model Comparison":
        st.header("Model Comparison Testing")
        
        # Show initialization status
        if selected_models:
            ready_models = [m for m in selected_models if m in st.session_state.chat_states]
            if ready_models:
                st.success(f"✅ **{len(ready_models)} models initialized and ready:** {', '.join(ready_models)}")
            else:
                st.warning("⚠️ **Models are being initialized...** Please wait.")
        else:
            st.info("📝 **Select models above to initialize them for comparison.**")
        
        st.markdown("---")  # Visual separator
        
        # Test query input
        test_query = st.text_input("Enter test query for model comparison:")
        test_name = st.text_input("Test name (optional):", placeholder="e.g., Error Analysis Test")
        
        # Option to use staggered requests (always visible)
        use_staggered = st.checkbox("Use staggered requests (recommended for rate limit avoidance)", value=True)
        
        # Show button only when there's a query and models are ready
        if test_query.strip() and selected_models and all(model in st.session_state.chat_states for model in selected_models):
            run_comparison = st.button("🚀 Run Comparison", type="primary")
        elif test_query.strip() and selected_models:
            st.warning("⚠️ **Waiting for models to initialize...** The comparison button will appear when ready.")
            run_comparison = False
        elif test_query.strip():
            st.info("💡 **Select models above to enable the comparison.**")
            run_comparison = False
        else:
            st.info("💡 **Enter a test query above to enable the comparison button.**")
            run_comparison = False
        
        # Only process when button is clicked
        if run_comparison and test_query.strip():
            if selected_models and all(model in st.session_state.chat_states for model in selected_models):
                
                with st.spinner("Running comparison across all models..."):
                    # Filter chat states for selected models
                    comparison_states = {
                        model: st.session_state.chat_states[model] 
                        for model in selected_models
                    }
                    
                    # Run model comparison (simple version without rate limit filtering)
                    results = run_model_comparison_smart(comparison_states, test_query, test_name)
                    
                    st.session_state.comparison_results.extend(results)
                
                # Display results
                st.subheader("Comparison Results")
                
                for result in results:
                    # Color code based on success/failure
                    if result['success']:
                        status_emoji = "✅"
                        time_display = f"{result['metrics']['total_time']}s"
                    elif result['error_type'] == 'rate_limit':
                        status_emoji = "⚠️"
                        time_display = "Rate Limited"
                    else:
                        status_emoji = "❌"
                        time_display = "Failed"
                    
                    with st.expander(f"{status_emoji} {result['model_name']} - {time_display}"):
                        if result['success']:
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write("**Response:**")
                                response_text = result.get('response', 'No response available')
                                if response_text is None:
                                    response_text = 'No response available'
                                st.write(response_text)
                            
                            with col2:
                                metrics = result['metrics']
                                st.metric("Total Time", f"{metrics['total_time']}s")
                                st.metric("LLM Time", f"{metrics['llm_time']}s")
                                st.metric("Response Length", f"{metrics['response_length']} chars")
                        else:
                            if result['error_type'] == 'rate_limit':
                                response_text = result.get('response', 'Rate limit reached')
                                if response_text is None:
                                    response_text = 'Rate limit reached - please wait before retrying'
                                
                                # Enhanced rate limit information
                                st.error(f"**⚠️ Rate Limit Hit**")
                                st.markdown(f"**Message:** {response_text}")
                                
                                # Extract detailed metrics if available
                                metrics = result.get('metrics', {})
                                if metrics:
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        if metrics.get('wait_time'):
                                            st.metric("Wait Time", f"{metrics['wait_time']:.1f}s")
                                        
                                        # Show usage information if available
                                        usage_info = metrics.get('usage_info', {})
                                        if usage_info.get('used') and usage_info.get('limit'):
                                            st.metric("Token Usage", f"{usage_info['used']}/{usage_info['limit']}")
                                        elif usage_info.get('limit'):
                                            st.metric("Rate Limit", f"{usage_info['limit']} tokens/min")
                                    
                                    with col2:
                                        if usage_info.get('requested'):
                                            st.metric("Tokens Requested", usage_info['requested'])
                                        
                                        # Calculate percentage if we have both used and limit
                                        if usage_info.get('used') and usage_info.get('limit'):
                                            percentage = (usage_info['used'] / usage_info['limit']) * 100
                                            st.metric("Usage Percentage", f"{percentage:.1f}%")
                                
                                # Enhanced suggestions
                                st.info("💡 **Suggestions:**")
                                suggestions = [
                                    "• Wait for the specified time before retrying",
                                    "• Try using 'Llama 3.1 8B Instant' model (lower resource usage)",
                                    "• Use staggered requests with longer delays",
                                    "• Reduce the length of your query"
                                ]
                                
                                if metrics.get('wait_time'):
                                    suggestions[0] = f"• Wait {metrics['wait_time']:.1f} seconds before retrying"
                                
                                for suggestion in suggestions:
                                    st.markdown(suggestion)
                                    
                            elif result['error_type'] == 'skipped':
                                response_text = result.get('response', 'Skipped to avoid rate limits')
                                if response_text is None:
                                    response_text = 'Skipped to avoid rate limits'
                                st.info(f"**Skipped:** {response_text}")
                            else:
                                response_text = result.get('response', 'Unknown error occurred')
                                if response_text is None:
                                    response_text = 'Unknown error occurred'
                                st.error(f"**Error:** {response_text}")
                
                # Performance comparison chart
                if any(r['success'] for r in results):
                    st.subheader("Performance Comparison")
                    
                    # Create performance DataFrame
                    perf_data = []
                    for result in results:
                        if result['success']:
                            perf_data.append({
                                'Model': result['model_name'],
                                'Total Time (s)': result['metrics']['total_time'],
                                'LLM Time (s)': result['metrics']['llm_time'],
                                'Retrieval Time (s)': result['metrics']['retrieval_time']
                            })
                    
                    df = pd.DataFrame(perf_data)
                    
                    # Bar chart for response times
                    fig = px.bar(df, x='Model', y='Total Time (s)', 
                               title='Response Time Comparison')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Please select models and ensure they are initialized.")
    
    elif testing_mode == "Performance Analysis":
        st.header("Performance Analysis Dashboard")
        
        if st.session_state.chat_states:
            # Model performance summaries
            st.subheader("Model Performance Summary")
            
            summary_data = []
            for model_name, chat_state in st.session_state.chat_states.items():
                summary = get_performance_summary(chat_state)
                if summary:
                    summary_data.append(summary)
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary)
                
                # Performance trends chart
                if len(summary_data) > 1:
                    fig = px.bar(df_summary, x='model_name', y='avg_total_time',
                               title='Average Response Time by Model')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Detailed comparison results
            if st.session_state.comparison_results:
                st.subheader("Historical Comparison Results")
                
                # Convert to DataFrame for analysis
                comparison_df = []
                for result in st.session_state.comparison_results:
                    if result['success']:
                        row = {
                            'Model': result['model_name'],
                            'Test Name': result['test_name'],
                            'Total Time': result['metrics']['total_time'],
                            'LLM Time': result['metrics']['llm_time'],
                            'Response Length': result['metrics']['response_length']
                        }
                        comparison_df.append(row)
                
                if comparison_df:
                    df_comp = pd.DataFrame(comparison_df)
                    st.dataframe(df_comp)
                    
                    # Download results
                    csv = df_comp.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Results CSV",
                        data=csv,
                        file_name="model_comparison_results.csv",
                        mime="text/csv"
                    )
        else:
            st.info("No performance data available. Run some queries first!")

# Clear results button
if st.session_state.comparison_results:
    if st.button("🗑️ Clear Results"):
        st.session_state.comparison_results = []
        st.rerun()
