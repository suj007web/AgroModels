from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import json
import torchvision.models as models
import torch.nn as nn
import joblib
import pickle
import numpy as np
app = Flask(__name__)

with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) 
        self.conv4 = ConvBlock(256, 512, pool=True)
        #self.conv5 = ConvBlock(256, 256, pool=True)
        #self.conv6 = ConvBlock(256, 512, pool=True)
        #self.conv7 = ConvBlock(512, 512, pool=True)
        
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, x): # x is the loaded batch
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        #out = self.conv5(out)
        #out = self.conv6(out)
        #out = self.conv7(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out        



# Load saved model
model = CNN_NeuralNet(3, len(class_labels))  
checkpoint = torch.load("plant_disease_model.pth", map_location="cpu")


model.load_state_dict(torch.load('plant_disease_model.pth', map_location=torch.device('cpu')))
model.eval()

with open ('yield_Prediction.pkl', 'rb') as f:
    yield_prediction_model = pickle.load(f)
    
model_yield = yield_prediction_model['model']
scaler_yield = yield_prediction_model['scaler']
le_season_yield = yield_prediction_model['le_season']
le_crop_yield = yield_prediction_model['le_crop']
    


try:
    crop_recommendation_model = joblib.load('crop_recommendation_model.pkl')
except:
    print("Warning: Could not load crop recommendation model")
    crop_recommendation_model = None



transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():

    image = Image.open(request.files['image'])
    image = transform(image).unsqueeze(0)  
    

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    

    return jsonify({'class': class_labels[predicted.item()]})


@app.route('/recommend-crop', methods=['POST'])
def recommend_crop():
    data = request.get_json()
    

    features = np.array([[
        float(data.get('nitrogen')),
        float(data.get('phosphorus')),
        float(data.get('potassium')),
        float(data.get('temperature')),
        float(data.get('humidity')),
        float(data.get('ph')),
        float(data.get('rainfall'))
    ]])
    

    if crop_recommendation_model is not None:
        prediction = crop_recommendation_model.predict(features)
        return jsonify({'recommended_crop': prediction[0]})
    else:
        return jsonify({'error': 'Model not available'}), 503
@app.route('/predict-yield', methods=['POST'])
def predict_yield():
    data = request.get_json()
    
    # Get inputs
    temperature = float(data.get('temperature'))
    humidity = float(data.get('humidity'))
    soil_moisture = float(data.get('soil_moisture'))
    area = float(data.get('area'))
    season = data.get('season')
    crop = data.get('crop')
    
    season_encoded = le_season_yield.transform([season])[0]
    crop_encoded = le_crop_yield.transform([crop])[0]

    # Create input array
    input_data = [[season_encoded, crop_encoded, temperature, 
                humidity, soil_moisture, area]]

    # Scale input
    input_scaled = scaler_yield.transform(input_data)

    # Make prediction
    prediction = model_yield.predict(input_scaled)[0]
    
    if prediction is not None:
        return jsonify({
            'predicted_production_tons': round(prediction, 2),
            'yield_per_acre': round(prediction/area, 2)
        })
    else:
        return jsonify({
            'error': f'No yield data available for crop {crop} in season {season}'
        }), 404
if __name__ == '__main__':
    app.run(debug=True)