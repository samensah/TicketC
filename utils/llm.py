"""
Simplified LLM integration for the PIA Analysis Tool.
"""

import json
import streamlit as st
import requests

# Import transformers for direct model loading - wrapped in try/except to handle missing dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class LLMIntegration:
    """
    Class for integrating with language models for summarization.
    """
    
    def __init__(self):
        self.endpoint_types = [
            "Direct Transformers",
            "Ollama",
            "Custom API"
        ]
        self.ollama_models = [
            "llama2",
            "mistral",
            "phi4-mini"
        ]
    
    def display_config_ui(self):
        """
        Display configuration UI for LLM integration.
        
        Returns:
            dict: Configuration parameters
        """
        st.subheader("LLM Configuration")
        
        # endpoint_type = st.selectbox(
        #     "LLM Endpoint Type",
        #     self.endpoint_types
        # )
        endpoint_type="Ollama"
        
        config = {"endpoint_type": endpoint_type}
        
        if endpoint_type == "Direct Transformers":
            if not TRANSFORMERS_AVAILABLE:
                st.warning("Transformers library not available. Please install with: pip install transformers torch")
            
            config["model_path"] = st.text_input(
                "Model Path", 
                value="/path/to/your/local/model",
                help="Path to your local model directory or Hugging Face model name"
            )
            # Automatically select cuda if available, otherwise cpu
            config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            st.info(f"Using device: {config['device']}")
            
            # Advanced parameters in expander
            with st.expander("Advanced Parameters"):
                config["temperature"] = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
                config["max_tokens"] = st.slider("Max New Tokens", 100, 2000, 800)
                config["top_p"] = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)
                config["top_k"] = st.slider("Top K", 1, 100, 50, 1)
                
        elif endpoint_type == "Ollama":
            # config["ollama_url"] = st.text_input("Ollama API URL", value="http://localhost:11434")
            config["ollama_url"] ="http://localhost:11434"
            config["model"] = st.selectbox("Select Model", self.ollama_models)
            
            # Basic parameters
            config["temperature"] = 0.1
            config["max_tokens"] = 2000
            # config["temperature"] = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
            # config["max_tokens"] = st.slider("Max Tokens", 100, 2000, 8000)
            
        elif endpoint_type == "Custom API":
            config["api_url"] = st.text_input("API URL", value="http://localhost:8000/generate")
            config["api_key"] = st.text_input("API Key (if required)", type="password")
            config["model"] = st.text_input("Model Name (if required)")
            config["max_tokens"] = st.slider("Max Tokens", 100, 2000, 800)
        
        return config
    
    def generate_summary(self, texts, field_name, llm_config):
        """
        Generate summary of texts using the configured LLM.
        
        Args:
            texts: List of texts to summarize
            field_name: Name of the field being summarized
            llm_config: LLM configuration parameters
            
        Returns:
            str: Generated summary
        """
        # Sample up to 10 texts for the prompt
        # sample_texts = texts[:10] if len(texts) > 10 else texts
        sample_texts = texts
        
        # Create the prompt
        prompt = f"""
        The following are {len(sample_texts)} examples of text.
        These texts are from the field '{field_name}'.
        
        EXAMPLES:
        
        {json.dumps(sample_texts, indent=2)}
        
        Please provide: a concise summary (3-5 sentences) of the common themes in these texts
        """
        
        endpoint_type = llm_config.get("endpoint_type")
        
        try:
            if endpoint_type == "Direct Transformers":
                return self._call_direct_transformers(prompt, llm_config)
            elif endpoint_type == "Ollama":
                return self._call_ollama(prompt, llm_config)
            elif endpoint_type == "Custom API":
                return self._call_custom_api(prompt, llm_config)
            else:
                return "Unsupported LLM endpoint type"
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def _call_direct_transformers(self, prompt, config):
        """
        Generate text directly using a local model with the Transformers library.
        
        Args:
            prompt: The input prompt
            config: Configuration parameters
        
        Returns:
            str: Generated text
        """
        if not TRANSFORMERS_AVAILABLE:
            return "Error: Transformers library not available. Please install with: pip install transformers torch"
        
        try:
            model_path = config.get("model_path")
            device = config.get("device", "cpu")
            max_tokens = config.get("max_tokens", 800)
            temperature = config.get("temperature", 0.7)
            top_k = config.get("top_k", 50)
            top_p = config.get("top_p", 0.9)
            
            # Status updates
            st.write("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            st.write("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move model to the specified device
            model = model.to(device)
            
            # Tokenize input
            st.write("Tokenizing input...")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate text
            st.write("Generating text...")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=len(inputs.input_ids[0]) + max_tokens,
                    do_sample=True,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    num_return_sequences=1,
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove input prompt)
            prompt_text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
            if generated_text.startswith(prompt_text):
                generated_text = generated_text[len(prompt_text):].strip()
            
            return generated_text
            
        except Exception as e:
            return f"Error using local model: {str(e)}"


    
    def _call_ollama(self, prompt, config):
        """
        Call Ollama API for text generation.
        
        Args:
            prompt: The input prompt
            config: Configuration parameters
            
        Returns:
            str: Generated text
        """
        url = config.get("ollama_url", "http://localhost:11434")
        model = config.get("model", "llama2")
        max_tokens = config.get("max_tokens", 800)
        temperature = config.get("temperature", 0.7)
        
        try:
            response = requests.post(
                f"{url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                },
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error calling Ollama API: {str(e)}"
    
    def _call_custom_api(self, prompt, config):
        """
        Call custom API for text generation.
        Placeholder for future implementation.
        
        Args:
            prompt: The input prompt
            config: Configuration parameters
            
        Returns:
            str: Generated text
        """
        # This is a placeholder for your custom API implementation
        api_url = config.get("api_url", "")
        api_key = config.get("api_key", "")
        model = config.get("model", "")
        max_tokens = config.get("max_tokens", 800)
        
        # Example implementation - modify according to your API's requirements
        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                
            response = requests.post(
                api_url,
                headers=headers,
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens
                }
            )
            
            if response.status_code == 200:
                # Adjust according to your API's response format
                return response.json().get("generated_text", "")
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error calling custom API: {str(e)}"
