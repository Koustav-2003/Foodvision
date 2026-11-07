# FoodVision – Food Image Classification

FoodVision is a deep learning-based food image classification application built using **PyTorch and ConvNeXt Small**. The model is fine-tuned on the Food-101 dataset to classify food images into 101 different categories.

The application uses **FastAPI** for backend model inference and **Streamlit** for an interactive user interface.

## 🚀 Live Application

**Try FoodVision:** [Live Demo – Coming Soon](#)

## Tech Stack

- **Deep Learning:** PyTorch, Torchvision
- **Model Architecture:** ConvNeXt Small
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **Image Processing:** Pillow
- **Dataset:** Food-101

## Model Overview

- **Architecture:** ConvNeXt Small
- **Pretrained Weights:** ImageNet-1K
- **Dataset:** Food-101
- **Number of Classes:** 101
- **Approach:** Transfer Learning and Fine-Tuning
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

The model was fine-tuned using PyTorch with data augmentation and evaluated on Food-101 images.

## Features

- Food image classification across 101 categories
- Deep learning inference using ConvNeXt Small
- REST API built with FastAPI
- Interactive image upload interface using Streamlit
- Prediction confidence scores

## Supported Food Categories

Apple Pie, Baby Back Ribs, Baklava, Beef Carpaccio, Beef Tartare, Beet Salad, Beignets, Bibimbap, Bread Pudding, Breakfast Burrito, Bruschetta, Caesar Salad, Cannoli, Caprese Salad, Carrot Cake, Ceviche, Cheese Plate, Cheesecake, Chicken Curry, Chicken Quesadilla, Chicken Wings, Chocolate Cake, Chocolate Mousse, Churros, Clam Chowder, Club Sandwich, Crab Cakes, Creme Brulee, Croque Madame, Cup Cakes, Deviled Eggs, Donuts, Dumplings, Edamame, Eggs Benedict, Escargots, Falafel, Filet Mignon, Fish and Chips, Foie Gras, French Fries, French Onion Soup, French Toast, Fried Calamari, Fried Rice, Frozen Yogurt, Garlic Bread, Gnocchi, Greek Salad, Grilled Cheese Sandwich, Grilled Salmon, Guacamole, Gyoza, Hamburger, Hot and Sour Soup, Hot Dog, Huevos Rancheros, Hummus, Ice Cream, Lasagna, Lobster Bisque, Lobster Roll Sandwich, Macaroni and Cheese, Macarons, Miso Soup, Mussels, Nachos, Omelette, Onion Rings, Oysters, Pad Thai, Paella, Pancakes, Panna Cotta, Peking Duck, Pho, Pizza, Pork Chop, Poutine, Prime Rib, Pulled Pork Sandwich, Ramen, Ravioli, Red Velvet Cake, Risotto, Samosa, Sashimi, Scallops, Seaweed Salad, Shrimp and Grits, Spaghetti Bolognese, Spaghetti Carbonara, Spring Rolls, Steak, Strawberry Shortcake, Sushi, Tacos, Takoyaki, Tiramisu, Tuna Tartare, Waffles.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Koustav-2003/Foodvision
cd FoodVision
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

### Start the FastAPI Backend

```bash
uvicorn backend.main:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

### Start the Streamlit Frontend

```bash
streamlit run frontend/app.py
```

*These commands will work once the application code is implemented.*

## Dataset

The model was trained using the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

- 101 food categories
- 101,000 images
- 75,750 training images
- 25,250 testing images

## Future Improvements

- Cloud deployment
- Real-world image evaluation
- Model inference optimization
- Improved confidence calibration and handling of unsupported food categories

## Author

Developed as an end-to-end deep learning project demonstrating transfer learning, computer vision, REST API development and model deployment.