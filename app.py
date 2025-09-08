import gradio as gr
from huggingface_hub import InferenceClient
import os



client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HF_TOKEN"),
)

def classify_image(image):
    """
    Classify an uploaded image as safe for work or not safe for work
    """
    if image is None:
        return "No image uploaded"
    
    try:
        # Use the NSFW classification model
        output = client.image_classification(image, model="Falconsai/nsfw_image_detection")
        
        # Find the highest confidence prediction
        if output:
            # Sort by confidence score (highest first)
            sorted_output = sorted(output, key=lambda x: x.score, reverse=True)
            top_prediction = sorted_output[0]
            
            # Determine if it's safe for work or not
            if top_prediction.label == 'normal':
                classification = "Safe for work"
            else:  # nsfw
                classification = "Not safe for work"
            
            confidence_percent = round(top_prediction.score * 100, 2)
            
            return f"{classification} ({confidence_percent}% confidence)"
        else:
            return "No classification result"
            
    except Exception as e:
        return f"Error: {str(e)}"

# Create the Gradio interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    title="NSFW Image Classification (Github Action works! v2)",
    description="Upload an image to check if it's safe for work or not safe for work"
)

if __name__ == "__main__":
    demo.launch()