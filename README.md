## Prototype Development for Image Captioning Using the BLIP Model and Gradio Framework

### AIM:
To design and deploy a prototype application for image captioning by utilizing the BLIP image-captioning model and integrating it with the Gradio UI framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Manual generation of descriptive text for digital images is resource-intensive and subjective, while standard computer vision pipelines often fail to capture complex contextual nuances, resulting in incomplete descriptions. Furthermore, state-of-the-art machine learning models typically remain confined to technical environments, lacking intuitive interfaces for non-technical users.

To address these limitations, this project focuses on developing and deploying an end-to-end prototype application. By utilizing the pre-trained BLIP (Bootstrapping Language-Image Pre-training) model via an automated Inference Endpoint, the system will reliably generate accurate, context-aware image captions. This backend is seamlessly integrated with the Gradio UI framework, providing a streamlined, web-accessible interface for instantaneous user interaction and model evaluation.

### DESIGN STEPS:

### STEP 1: Environment Setup and API Authentication
Initialize the workspace environment by installing and importing essential libraries (gradio, requests, PIL, dotenv).

Securely load the Hugging Face API credentials from a local .env configuration file to authenticate requests against the hosted inference models.

### STEP 2: Backend Logic and Inference Pipeline
Develop a core helper function (get_completion) to transmit authenticated HTTP POST requests containing payload data to the Hugging Face Image-to-Text inference endpoint.

Implement a preprocessing utility to convert incoming PIL image objects into Base64-encoded strings, matching the serialization format expected by the remote BLIP API endpoint.

Construct the central prediction function (captioner) to orchestrate image conversion, endpoint execution, and response parsing to extract the generated text.

### STEP 3: UI Design and Application Deployment
Instantiate a web application interface using gr.Interface(), defining an image uploader component restricted to PIL format as the input and a text field for displaying the resulting caption as the output.

Configure application metadata including customized titles, descriptions, and local image examples to improve usability.

Safely terminate any lingering active network sockets via gr.close_all() and launch the interactive server instance on a designated port with a shareable public URL enabled.
### PROGRAM:
~~~
import base64
import io
import json
import os
import requests
from dotenv import find_dotenv, load_dotenv
import gradio as gr
import IPython.display
from PIL import Image

# 1. Load your HF API key and environment variables
_ = load_dotenv(find_dotenv())  # read local .env file
hf_api_key = os.environ["HF_API_KEY"]


# 2. Define the Helper function for the Image-to-text endpoint
def get_completion(
    inputs, parameters=None, ENDPOINT_URL=os.environ["HF_API_ITT_BASE"]
):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json",
    }
    data = {"inputs": inputs}
    if parameters is not None:
        data.update({"parameters": parameters})

    response = requests.request(
        "POST", ENDPOINT_URL, headers=headers, data=json.dumps(data)
    )
    return json.loads(response.content.decode("utf-8"))


# 3. Define the captioner app logic
def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format="PNG")
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode("utf-8"))


def captioner(image):
    base64_image = image_to_base64_str(image)
    result = get_completion(base64_image)
    return result[0]["generated_text"]


# 4. Clear any existing connections and launch the Gradio Interface
gr.close_all()

demo = gr.Interface(
    fn=captioner,
    inputs=[gr.Image(label="Upload image", type="pil")],
    outputs=[gr.Textbox(label="Caption")],
    title="Image Captioning with BLIP",
    description="Caption any image using the BLIP model",
    allow_flagging="never",
    examples=["christmas_dog.jpeg", "bird_flight.jpeg", "cow.jpeg"],
)

demo.launch(share=True, server_port=int(os.environ["PORT1"]))
~~~

### OUTPUT:

<img width="945" height="618" alt="image" src="https://github.com/user-attachments/assets/6fcc37f9-d1c2-443c-bfbd-3a47d3e21ef4" />


### RESULT:

Thus the program has been executed successfully.
