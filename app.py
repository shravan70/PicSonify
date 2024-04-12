from flask import Flask, render_template, request, send_file
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import gtts
import os
import uuid

app = Flask(__name__)

# Define model and related components
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Parameters for text generation
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Define a function to handle prediction and audio generation
def predict_and_generate_audio(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)

    # Generate caption for the image
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    prediction = preds[0]

    # Generate unique filename for the audio
    audio_filename = f"generated_audio_{uuid.uuid4().hex}.mp3"
    audio_path = os.path.join("Sound", audio_filename)

    # Generate audio from the caption
    sound = gtts.gTTS(text=prediction, lang="en")
    sound.save(audio_path)
    return prediction, audio_path
@app.route('/', methods=['GET', 'POST'])
def process_image():
    prediction = None
    audio_path = None

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return "No file part"

        imagefile = request.files['imagefile']
        if imagefile.filename == '':
            return "No selected file"

        imagepath = os.path.join("./images", imagefile.filename)
        imagefile.save(imagepath)

        # Perform prediction and audio generation
        prediction, audio_path = predict_and_generate_audio(imagepath)

    return render_template('index.html', prediction=prediction, audio_path=audio_path)
@app.route('/get_audio')
def get_audio():
    audio_path = request.args.get('audio_path')
    return send_file(audio_path, as_attachment=True)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
