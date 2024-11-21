import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import gzip
from torchvision.models import resnet18
import torch.nn.functional as F
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1]) 
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        x = F.normalize(x, dim=1)
        return x

def preprocess_image(image):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
    ])
    image = test_transform(image).unsqueeze(0)  # Add batch dimension
    return image

def assign_class_to_image(test_embedding, mean_embeddings):
    similarities = cosine_similarity([test_embedding], mean_embeddings)
    best_class = np.argmax(similarities)  # Class with the highest similarity
    return best_class

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device('cpu')
model = CNN().to(device)
model_save_path = "model_weights_cpu.pth"
model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
model = model.to(device)
model.eval()
print("Model weights loaded successfully.")

with gzip.open('mean_embeddings.pkl.gz', 'rb') as f:
    mean_embeddings = pickle.load(f)

name_mapper={0:'ajwain',1:'aswagandha',2:'black pepper',3:'cassia fistula',4:'elaichi',5:'gadhur',6:'ixora coccinea',7:'kalanchoe pinata',8:'LAGERSTROEMIA',9:'lawang',10:'MAGNOLIA CHAMPACA',11:'mimosa',12:'neem',13:'S6[unk]',14:'S7[unk]',15:'spathodea',16:'betel',17:'centella asiatica',18:'merremia_peltata'}
def main():
    st.title("Image Classification with Streamlit")
    
    # Upload image
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess the image
        image_tensor = preprocess_image(image)

        # Move the image to the device
        # if torch.cuda.is_available():
        #     image_tensor = image_tensor.cuda()

        # Get the model's output
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output = model(image_tensor)
        
        # Print the output shape to debug
        

        # Split output based on shape
        f1 = output[0]  # Batch size
        print(len(f1))
        # Compute the embedding for the image
        test_embedding = f1.cpu().numpy().flatten()  # Flatten to a 1D vector
        print(test_embedding)
        

        # Assign class using cosine similarity
        assigned_class = assign_class_to_image(test_embedding, mean_embeddings)

        # Show the predicted class
        st.write(f"Predicted Class: {name_mapper[assigned_class]}")


if __name__ == "__main__":
    main()