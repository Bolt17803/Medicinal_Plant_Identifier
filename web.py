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
import requests
from bs4 import BeautifulSoup
from googlesearch import search

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

def get_medicinal_info_from_google(plant_name):
    query = f"{plant_name} medicinal uses"
    # Use start and stop to get the top result (start=0 means the first page of results)
    results = list(search(query, num_results=2))  # This will fetch the top 1 result

    if results:
        return results[0]  # Return the URL of the top search result
    else:
        return "No results found."

def fetch_webpage(link):
    # Set custom headers to mimic a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    response = requests.get(link, headers=headers)

    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch the webpage. Status code: {response.status_code}")
        return None

# def extract_medical_info(page_content):
#     # Parse the HTML content
#     soup = BeautifulSoup(page_content, 'html.parser')

#     # Define common medical keywords
#     medical_keywords = ["health", "medication", "treatment", "disease", "side effect", "symptoms", "doctor", "cure", "prescription"]

#     # Extract all text from the page
#     text = soup.get_text()

#     # Filter text based on medical keywords
#     relevant_text = []
#     for line in text.split('\n'):
#         if any(keyword.lower() in line.lower() for keyword in medical_keywords):
#             relevant_text.append(line.strip())

#     # Return cleaned-up relevant content
#     return "\n".join(relevant_text)
def extract_medical_info(page_content):
    from collections import defaultdict

    # Parse the HTML content
    soup = BeautifulSoup(page_content, 'html.parser')

    # Define common medical keywords
    medical_keywords = ["health", "medication", "treatment", "disease", "side effect", "symptoms", "doctor", "cure", "prescription"]

    # Store results as a dictionary of subheadings and their corresponding text
    medical_info = defaultdict(list)

    # Identify heading tags and their corresponding content
    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        # Extract heading text
        heading_text = heading.get_text(strip=True)

        # Find the next sibling elements to gather the associated content
        content = []
        sibling = heading.find_next_sibling()

        while sibling and sibling.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if sibling.name in ['p', 'ul', 'ol', 'div'] and sibling.get_text(strip=True):
                content.append(sibling.get_text(strip=True))
            sibling = sibling.find_next_sibling()

        # Filter content based on medical keywords
        filtered_content = [
            line for line in content if any(keyword.lower() in line.lower() for keyword in medical_keywords)
        ]

        # Store the heading and its relevant content
        if filtered_content:
            medical_info[heading_text].extend(filtered_content)

    # Format the output as a string with headings and text
    output = ""
    for heading, texts in medical_info.items():
        output += f"\n{heading}:\n"
        output += "\n".join(f"- {text}" for text in texts)
        output += "\n"

    return output.strip()


def trigger(plant_name):
    link = get_medicinal_info_from_google(plant_name)

    if link:
        print(f"Top Google result: {link}")
        page_content = fetch_webpage(link)

        if page_content:
            medical_info = extract_medical_info(page_content)

            if medical_info:
                print("got few text")
                return medical_info
            else:
                return "No relevant medical information found on the page."
        else:
            return "Failed to fetch or parse the webpage."
    else:
        return "No search results found."

def get_data(plant_name):
    link=get_medicinal_info_from_google(plant_name)
    print(link)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    response = requests.get(link, headers=headers)
    if response.status_code == 200:
        page_content = response.text
        print("Page fetched successfully!")
        soup = BeautifulSoup(page_content, "html.parser")
    else:
        print(f"Failed to fetch the webpage. Status code: {response.status_code}")

    uses_component = soup.find('div', class_='uses-container')

    if uses_component:
        t1=uses_component.get_text(strip=True)
    else:
        t1=None
        print("The uses web component was not found.")
    
    side_effects_component = soup.find('div', class_='side-effects-page')

    if side_effects_component:
        t2=side_effects_component.get_text(strip=True)
    else:
        t2=None
        print("The uses web component was not found.")

    precautions_component = soup.find('div', class_='precautions-page')

    if precautions_component:
        t3=precautions_component.get_text(strip=True)
    else:
        t3=None
        print("The uses web component was not found.")

    return link,t1,t2,t3

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

        link, uses, side_effects, precautions = get_data(name_mapper[assigned_class])
        
        if uses!=None:
            st.write(f"Plants Uses: {uses}")
        if side_effects!=None:
            st.write(f"Side-Effects: {side_effects}")
        if precautions!=None:
            st.write(f"Precautions: {precautions}")

        st.write(f'Check this link for more plant info:{link}')

        info = trigger(name_mapper[assigned_class])
        st.write(f'Few Features of the Plant: \n {info}')



if __name__ == "__main__":
    main()