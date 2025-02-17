import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from agno.agent import Agent
from agno.media import Video
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv
from google import genai

# load_dotenv()

st.set_page_config(page_title="Video Analysis AI Agent", page_icon="🤖", layout="wide")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Enter your Google API Key",
        type="password",
        help="Enter your Google API key to use the service",
    )

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.markdown(
        """
        To use this application, you'll need to set up your API keys:
        
        1. **Google Gemini API**:
           - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
           - Create a new API key
           - Paste the API key in the input box above
        
        2. **DuckDuckGo API**:
           - No API key needed! 🎉
        """
    )

st.title("📹🤖 Video Analysis AI Agent")


# Initialize single agent with both capabilities
@st.cache_resource
def initialize_agent(model):
    return Agent(
        name="Multimodal Analyst",
        model=model,
        tools=[DuckDuckGoTools()],
        markdown=True,
    )


if not api_key:
    st.info("Please enter your Google API key in the sidebar to begin.")
else:
    model = Gemini(id="gemini-2.0-flash")
    agent = initialize_agent(model)

    # File uploader
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        st.video(video_path)

        user_prompt = st.text_area(
            "What would you like to know?",
            placeholder="Ask any question related to the video - the AI Agent will analyze it and search the web if needed",
            help="You can ask questions about the video content and get relevant information from the web",
        )

        if st.button("Analyze & Research"):
            if not user_prompt:
                st.warning("Please enter your question.")
            else:
                try:
                    with st.spinner("Processing video and researching..."):
                        video_file = model.get_client().files.upload(file=video_path)
                        while video_file.state.name == "PROCESSING":
                            time.sleep(2)
                            video_file = model.get_client().files.get(
                                name=video_file.name
                            )

                        prompt = f"""
                        First analyze this video and then answer the following question using both 
                        the video analysis and web research: {user_prompt}
                        
                        Provide a comprehensive response focusing on practical, actionable information.
                        """

                        result = agent.run(prompt, videos=[Video(content=video_file)])

                    st.subheader("Result")
                    st.markdown(result.content)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    Path(video_path).unlink(missing_ok=True)
    else:
        st.info("Please upload a video to begin analysis.")

    st.markdown(
        """
        <style>
        .stTextArea textarea {
            height: 100px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
