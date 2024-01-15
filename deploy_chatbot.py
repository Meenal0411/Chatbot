import streamlit as st
import replicate
import os
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer

# Global variables for model and tokenizer
gpt2_model = None
gpt2_tokenizer = None

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.1
    if "top_p" not in st.session_state:
        st.session_state.top_p = 0.9
    if "max_length" not in st.session_state:
        st.session_state.max_length = 120

# Function to generate LLaMA2 response
@st.cache_data(show_spinner=False)
def load_gpt2_model_and_tokenizer():
    global gpt2_model, gpt2_tokenizer
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to generate LLaMA2 response
@st.cache_data(show_spinner=False)
def generate_llama2_response_cached(prompt_input, string_dialogue, llm, temperature, top_p, max_length, messages):
    for dict_message in messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    # Call the actual function without await
    output = replicate.run(llm, {
        "prompt": f"{string_dialogue} {prompt_input} Assistant: ",
        "temperature": temperature,
        "top_p": top_p,
        "max_length": max_length,
        "repetition_penalty": 1
    })

    return list(output)

# Function to generate LLaMA2 response
def generate_llama2_response(prompt_input, string_dialogue, llm, temperature, top_p, max_length, messages):
    return generate_llama2_response_cached(prompt_input, string_dialogue, llm, temperature, top_p, max_length, messages)

# Main function
def main():
    # Set Replicate API token
    replicate_api = "r8_UT1Zofte3rCH5CzEYooyEtMgUxZKDbY1lLn3P"
    os.environ["REPLICATE_API_TOKEN"] = replicate_api

    # Set page configuration
    st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

    # Initialize session state
    init_session_state()

    # Load GPT-2 model and tokenizer
    if gpt2_model is None or gpt2_tokenizer is None:
        load_gpt2_model_and_tokenizer()

    # Replicate Credentials
    with st.sidebar:
        st.title('🦙💬 Chatbot Menu')
        st.write('Choose a chatbot to interact with.')

        option = st.selectbox("Select a Chatbot", ["Home", "Llama 2", "About Us"])

    if option == "Home":
        col1, col2 = st.columns([2, 1])  # Adjust the ratio based on your preference
        with col1:
            st.write("## Welcome to the Chatbot")
            st.write("Feel free to interact with the chatbot.")
            st.markdown("<style>div.Widget.row-widget.stButton > button{margin-left:auto;margin-right:0}</style>", unsafe_allow_html=True)

       

    elif option == "Llama 2":
        st.subheader('Chatbot')
        # Only include Llama2-13B in the dropdown
        selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-13B'], key='selected_model')
        st.session_state.llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
        st.session_state.temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
        st.session_state.top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        st.session_state.max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
        # st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')
        # st.write("This chatbot is powered by Hugging Face's transformers library. It uses the Llama 2-13B language model, which is designed to provide intelligent and contextually relevant responses in natural language conversations.")
        # st.write("Feel free to interact with the chatbot on the 'Home' page!")

    # About Us page content
    if option == "About Us":
        st.markdown("## About Us")
        # Show specific content when Llama2-13B is selected
        if st.session_state.llm == 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5':
            st.write("This chatbot is powered by Hugging Face's transformers library. It utilizes the Llama 2-13B language model, which is designed to provide intelligent and contextually relevant responses in natural language conversations.")
            st.write("Llama 2-13B is a large language model developed by a16z Infra. It is trained on diverse and extensive datasets to understand and generate human-like text. The model is capable of handling various types of conversational queries and providing informative responses.")
    else:
        # Display or clear chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        def clear_chat_history():
            st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

        if st.sidebar.button('Clear Chat History', on_click=clear_chat_history):
            pass  # This should be indented inside the if statement

        # User-provided prompt
        if option == "Llama 2":
            if prompt_input := st.chat_input(disabled=not replicate_api):
                st.session_state.messages.append({"role": "user", "content": prompt_input})
                with st.chat_message("user"):
                    st.write(prompt_input)

                # Generate a new response if the last message is not from the assistant
                if st.session_state.messages[-1]["role"] != "assistant":
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = generate_llama2_response(prompt_input, "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.", st.session_state.llm, st.session_state.temperature, st.session_state.top_p, st.session_state.max_length, st.session_state.messages)
                            placeholder = st.empty()
                            full_response = ''
                            for item in response:
                                full_response += item
                                placeholder.markdown(full_response)
                            placeholder.markdown(full_response)

if __name__ == "__main__":
    main()
