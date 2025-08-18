import streamlit as st
import os
import tempfile
from PIL import Image
import io
import base64

from dotenv import load_dotenv
load_dotenv()

# Import your existing functions
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

# System prompt (same as in your gradio app)
system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            What's in this image?. Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""

def save_uploaded_image(uploaded_file):
    """Save uploaded image to a temporary file and return the path"""
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    return None

def save_audio_file(audio_bytes):
    """Save audio bytes to a temporary file and return the path"""
    if audio_bytes is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            return tmp_file.name
    return None

def process_inputs(audio_filepath, image_filepath):
    """Process audio and image inputs (same logic as your gradio app)"""
    speech_to_text_output = ""
    
    # Process audio if provided
    if audio_filepath and os.path.exists(audio_filepath):
        try:
            speech_to_text_output = transcribe_with_groq(
                GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
                audio_filepath=audio_filepath,
                stt_model="whisper-large-v3"
            )
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")
            speech_to_text_output = "Error transcribing audio"

    # Process image if provided
    if image_filepath and os.path.exists(image_filepath):
        try:
            encoded_image = encode_image(image_filepath)
            doctor_response = analyze_image_with_query(
                query=system_prompt + speech_to_text_output, 
                encoded_image=encoded_image, 
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
        except Exception as e:
            st.error(f"Error analyzing image: {e}")
            doctor_response = "Error analyzing image"
    else:
        doctor_response = "No image provided for me to analyze"

    # Generate voice response
    voice_filepath = None
    if doctor_response and doctor_response != "No image provided for me to analyze":
        try:
            voice_filepath = "doctor_response.mp3"
            text_to_speech_with_elevenlabs(
                input_text=doctor_response, 
                output_filepath=voice_filepath
            )
        except Exception as e:
            st.error(f"Error generating voice response: {e}")
            try:
                # Fallback to gTTS
                text_to_speech_with_gtts(
                    input_text=doctor_response, 
                    output_filepath=voice_filepath
                )
            except Exception as e2:
                st.error(f"Error with fallback TTS: {e2}")
                voice_filepath = None

    return speech_to_text_output, doctor_response, voice_filepath

def main():
    st.set_page_config(
        page_title="AI Doctor with Vision and Voice",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 AI Doctor with Vision and Voice")
    st.markdown("---")

    # Input Section at the top
    st.header("📝 Input Section")
    
    # Audio Input Section
    st.subheader("🎤 Audio Input")
    audio_filepath = None
    
    st.info("Click the button below to start recording")
    if st.button("🎤 Start Recording"):
        with st.spinner("Recording... Speak now!"):
            try:
                audio_filepath = "recorded_audio.mp3"
                record_audio(file_path=audio_filepath, timeout=10, phrase_time_limit=30)
                st.success("Recording completed!")
                # Store audio filepath in session state
                st.session_state.audio_filepath = audio_filepath
            except Exception as e:
                st.error(f"Recording failed: {e}")

    # Image Input Section
    st.subheader("📸 Image Input")
    image_input_method = st.radio(
        "Choose image input method:",
        ["Upload Image", "Take Photo"]
    )
    
    image_filepath = None
    
    if image_input_method == "Upload Image":
        uploaded_image = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        if uploaded_image is not None:
            image_filepath = save_uploaded_image(uploaded_image)
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            # Store image filepath in session state
            st.session_state.image_filepath = image_filepath
            
    elif image_input_method == "Take Photo":
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            image_filepath = save_uploaded_image(camera_image)
            st.success("Photo captured successfully!")
            # Store image filepath in session state
            st.session_state.image_filepath = image_filepath

    # Process Button
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        # Get filepaths from session state if available
        current_audio = getattr(st.session_state, 'audio_filepath', None)
        current_image = getattr(st.session_state, 'image_filepath', None)
        
        if not current_audio and not current_image:
            st.warning("Please provide either audio input or image input (or both)")
        else:
            with st.spinner("Processing your request..."):
                speech_to_text_output, doctor_response, voice_filepath = process_inputs(
                    current_audio, current_image
                )
                
                # Store results in session state
                st.session_state.speech_to_text = speech_to_text_output
                st.session_state.doctor_response = doctor_response
                st.session_state.voice_filepath = voice_filepath

    st.markdown("---")

    # Results Section
    st.header("📋 **Results Section**")
    
    # Display Speech to Text Output
    if hasattr(st.session_state, 'speech_to_text') and st.session_state.speech_to_text:
        st.subheader("🎯 Speech to Text")
        st.text_area(
            "Transcribed Text:",
            value=st.session_state.speech_to_text,
            height=100,
            disabled=True
        )

    # Display Doctor's Response
    if hasattr(st.session_state, 'doctor_response') and st.session_state.doctor_response:
        st.subheader("👨‍⚕️ Doctor's Response")
        st.text_area(
            "Medical Analysis:",
            value=st.session_state.doctor_response,
            height=150,
            disabled=True
        )

    # Display Audio Response
    if hasattr(st.session_state, 'voice_filepath') and st.session_state.voice_filepath:
        if os.path.exists(st.session_state.voice_filepath):
            st.subheader("🔊 Voice Response")
            with open(st.session_state.voice_filepath, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/mp3')
                
                # Download button for audio
                st.download_button(
                    label="📥 Download Audio Response",
                    data=audio_bytes,
                    file_name="doctor_response.mp3",
                    mime="audio/mp3"
                )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>⚠️ <strong>Disclaimer:</strong> This is for educational purposes only. Always consult with a qualified healthcare professional for medical advice.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Cleanup temporary files on app restart
    if st.button("🧹 Clear Cache", help="Clear temporary files and reset the app"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

if __name__ == "__main__":
    main()