# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

#Step1: Setup GROQ API key
import os

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")

#Step2: Convert image to required format
import base64


#image_path="acne.jpg"

def encode_image(image_path):   
    image_file=open(image_path, "rb")
    return base64.b64encode(image_file.read()).decode('utf-8')

#Step3: Setup Multimodal LLM 
from groq import Groq

query = (
    "Look at the image and identify the injury. "
    "Reply only with the immediate emergency actions to take. "
    "Do not explain or describe symptoms. "
    "Do not include extra text, only list the steps."
)
#model = "meta-llama/llama-4-maverick-17b-128e-instruct"
model="meta-llama/llama-4-scout-17b-16e-instruct"
#model = "meta-llama/llama-4-scout-17b-16e-instruct"
#model="llama-3.2-90b-vision-preview" #Deprecated

def analyze_image_with_query(query, model, encoded_image):
    client = Groq()  
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )

    response_text = chat_completion.choices[0].message.content

    # Remove unwanted intro phrases if they exist
    unwanted_start = "With what I see, I think you have"
    if response_text.startswith(unwanted_start):
        # Find the first sentence break after the intro
        first_period = response_text.find(".")
        if first_period != -1:
            response_text = response_text[first_period + 1:].strip()

    return response_text
