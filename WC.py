import streamlit as st
import cv2
import base64
import time
import os
from dotenv import load_dotenv
from groq import Groq

class WasteClassifier:
    VALID_CATEGORIES = [
        'Recycling', 'Organic', 'Trash', 'Electronics', 'Miscellaneous',
        'Plastic', 'Metal', 'Glass', 'Paper', 'Textiles', 'Batteries',
        'Hazardous Waste', 'Food Waste', 'Bulky Waste', 'E-waste', 'Toxic Waste'
    ]

    def __init__(self, api_key=None):
        load_dotenv()
        
        # Use environment variable or passed API key
        api_key = api_key or os.getenv('GROQ_API_KEY')
        if not api_key:
            st.error("No Groq API key found. Please set GROQ_API_KEY in .env file.")
            raise ValueError("No API key provided")
        
        self.client = Groq(api_key=api_key)

    @staticmethod
    def encode_image(frame) -> str:
        """
        Convert OpenCV frame to base64 encoded image.
        """
        _, encoded_image = cv2.imencode('.jpg', frame)
        return base64.b64encode(encoded_image.tobytes()).decode('utf-8')

    def clean_response(self, response: str) -> tuple:
        """
        Clean and extract classification and handling advice.
        """
        # Remove any leading numbers or dots
        cleaned_lines = [line.strip().lstrip('0123456789. ') for line in response.split('\n')]
        
        # Ensure we have at least two lines
        if len(cleaned_lines) >= 2:
            return cleaned_lines[0], cleaned_lines[1]
        return "Miscellaneous", "General waste disposal recommended"

    def classify_image(self, base64_image: str) -> tuple:
        """
        Classify image and provide handling recommendations using vision model.
        Returns (classification, handling_advice)
        """
        if not base64_image:
            st.error("Empty image input")
            return "Miscellaneous", "General waste disposal recommended"

        try:
            response = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
                            EXPERT WASTE CLASSIFICATION TASK: Analyze the image and follow these STRICT guidelines:
                            CLASSIFICATION REQUIREMENTS:
                            - SELECT EXACTLY ONE waste category from this PRECISE list: {', '.join(self.VALID_CATEGORIES)}
                            - Base classification on VISUAL CHARACTERISTICS and MATERIAL COMPOSITION
                            - MOST SPECIFIC category takes precedence
                            - IGNORE context, focus SOLELY on the object's inherent waste type

                            RESPONSE FORMAT (MANDATORY):
                            - FIRST LINE: Waste Category (ONE WORD)
                            - SECOND LINE: Concise, actionable disposal recommendation
                            - NO additional text, numbering, or explanatory content
                            - USE clear, professional language
                            - PRIORITIZE local, environmentally responsible disposal methods

                            CRITICAL CONSTRAINTS:
                            - Maximum 10 words for category
                            - Maximum 40 words for disposal advice
                            - ZERO speculation or unnecessary details
                            """
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                max_tokens=100
            )
            
            full_response = response.choices[0].message.content.strip()
            print("full_response .......................", full_response)
            return self.clean_response(full_response)
        
        except Exception as e:
            st.error(f"Classification error: {e}")
            return "Miscellaneous", "General waste disposal recommended"

def main():
    st.set_page_config(page_title="Waste Classifier", page_icon=":recycle:")
    st.title("🌍 Real-Time Waste Classification")
    
    try:
        classifier = WasteClassifier()
    except ValueError:
        st.stop()

    # Initialize session state
    if 'classification' not in st.session_state:
        st.session_state.classification = None
    if 'handling_advice' not in st.session_state:
        st.session_state.handling_advice = None
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access webcam")
        return

    frame_placeholder = st.empty()
    result_placeholder = st.empty()
    advice_placeholder = st.empty()

    # Buttons outside the loop to prevent recreation
    classify_button = st.button("Classify Current Frame")
    stop_button = st.button("Stop Webcam")

    # Capture and display initial frame
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        st.session_state.current_frame = frame

    # Classification logic
    if classify_button and st.session_state.current_frame is not None:
        base64_image = WasteClassifier.encode_image(st.session_state.current_frame)
        st.session_state.classification, st.session_state.handling_advice = classifier.classify_image(base64_image)
        
        # Display classification and handling advice
        result_placeholder.success(f"Best Case Classification: {st.session_state.classification}")
        advice_placeholder.info(f"Handling Advice: {st.session_state.handling_advice}")

    # Stop webcam
    if stop_button:
        cap.release()
        st.warning("Webcam feed stopped")

if __name__ == "__main__":
    main()