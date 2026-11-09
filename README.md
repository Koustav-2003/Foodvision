# FoodVision – Food Image Classification

FoodVision is an end-to-end deep learning application built using **PyTorch and ConvNeXt Small** to classify food images into 101 different categories.

The model is fine-tuned on the Food-101 dataset using transfer learning. The application uses **FastAPI** for backend inference and **Streamlit** for an interactive frontend, allowing users to upload food images and receive predictions with confidence scores.

## Tech Stack

- **Deep Learning:** PyTorch, Torchvision
- **Model Architecture:** ConvNeXt Small
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **Image Processing:** Pillow
- **Dataset:** Food-101

## Model Overview

| Attribute | Details |
|---|---|
| Architecture | ConvNeXt Small |
| Pretrained Weights | ImageNet-1K |
| Dataset | Food-101 |
| Number of Classes | 101 |
| Approach | Transfer Learning and Fine-Tuning |
| Test Accuracy | Approximately 85% |

The model was fine-tuned using PyTorch with data augmentation and evaluated on the Food-101 test dataset.

Evaluation included accuracy, precision, recall, F1-score and confusion matrix analysis.

## Features

- Food image classification across 101 categories.
- Fine-tuned ConvNeXt Small for image recognition.
- REST API for model inference using FastAPI.
- Interactive image upload interface using Streamlit.
- Top-5 food predictions with confidence scores.
- Local model inference without requiring a cloud-hosted backend.

## Supported Food Categories

Apple Pie, Baby Back Ribs, Baklava, Beef Carpaccio, Beef Tartare, Beet Salad, Beignets, Bibimbap, Bread Pudding, Breakfast Burrito, Bruschetta, Caesar Salad, Cannoli, Caprese Salad, Carrot Cake, Ceviche, Cheese Plate, Cheesecake, Chicken Curry, Chicken Quesadilla, Chicken Wings, Chocolate Cake, Chocolate Mousse, Churros, Clam Chowder, Club Sandwich, Crab Cakes, Creme Brulee, Croque Madame, Cup Cakes, Deviled Eggs, Donuts, Dumplings, Edamame, Eggs Benedict, Escargots, Falafel, Filet Mignon, Fish and Chips, Foie Gras, French Fries, French Onion Soup, French Toast, Fried Calamari, Fried Rice, Frozen Yogurt, Garlic Bread, Gnocchi, Greek Salad, Grilled Cheese Sandwich, Grilled Salmon, Guacamole, Gyoza, Hamburger, Hot and Sour Soup, Hot Dog, Huevos Rancheros, Hummus, Ice Cream, Lasagna, Lobster Bisque, Lobster Roll Sandwich, Macaroni and Cheese, Macarons, Miso Soup, Mussels, Nachos, Omelette, Onion Rings, Oysters, Pad Thai, Paella, Pancakes, Panna Cotta, Peking Duck, Pho, Pizza, Pork Chop, Poutine, Prime Rib, Pulled Pork Sandwich, Ramen, Ravioli, Red Velvet Cake, Risotto, Samosa, Sashimi, Scallops, Seaweed Salad, Shrimp and Grits, Spaghetti Bolognese, Spaghetti Carbonara, Spring Rolls, Steak, Strawberry Shortcake, Sushi, Tacos, Takoyaki, Tiramisu, Tuna Tartare, Waffles.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Koustav-2003/Foodvision.git
cd Foodvision
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the appropriate CUDA-enabled PyTorch and Torchvision versions from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Add the Model Checkpoint

The trained model checkpoint is not included in the GitHub repository because of its file size.

Download `convnext_final.pth` from the [FoodVision Hugging Face model repository](https://huggingface.co/Koustav-717/foodvision-convnext-small) and place it inside the `models` directory.

Your directory should look like:

```text
Foodvision/
│
├── backend/
│   ├── main.py
│   ├── model.py
│   └── classes.py
│
├── frontend/
│   └── app.py
│
├── models/
│   └── convnext_final.pth
│
└── requirements.txt
```

## Running the Application

The application runs locally with the FastAPI backend and Streamlit frontend operating separately.

### 1. Start the FastAPI Backend

From the project root, run:

```bash
uvicorn backend.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 2. Start the Streamlit Frontend

Open another terminal, activate the same virtual environment, and run:

```bash
streamlit run frontend/app.py
```

The application will open in your browser.

## Dataset

The model was trained using the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

- 101 food categories
- 101,000 images
- 75,750 training images
- 25,250 testing images

## Future Improvements

- Cloud deployment.
- Real-world image evaluation.
- Model inference optimization.
- Improved confidence calibration.
- Handling of unsupported food categories.

## Author

Developed as an end-to-end deep learning project demonstrating transfer learning, computer vision, model fine-tuning, REST API development and interactive application development.