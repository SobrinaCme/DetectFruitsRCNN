import streamlit as st
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import gdown

# Load model
@st.cache_resource
def load_model():

    model_path = "model.pth"
    # Download from Google Drive if not exists
    if not os.path.exists(model_path):
        # Replace this with your real file ID
        file_id = "17t9AQZ1i7IL5azoZvym9HqfQ_yFiQGiY"
        url = f"https://drive.google.com/uc?id={file_id}"
        st.write("⬇️ Downloading model weights...")
        gdown.download(url, model_path, quiet=False)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Replace classifier head
    num_classes = 4  # 3 fruits + 1 background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load saved weights
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Label map
label_map = {1: 'apple', 2: 'banana', 3: 'orange'}

# Predict
def predict(image):
    image_tensor = F.to_tensor(image)
    with torch.no_grad():
        output = model([image_tensor])[0]
    return output

# Draw boxes
def draw_boxes(image, output):
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
        if score > 0.5:
            box = [round(i) for i in box.tolist()]
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1]), f"{label_map[label.item()]}: {score:.2f}", fill="red")
    return image

# Streamlit UI
st.title("🍎🍌🍊 Fruit Detector App")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🔍 Detecting...")
    output = predict(image)
    result_image = draw_boxes(image.copy(), output)
    st.image(result_image, caption="Detection Result", use_column_width=True)
