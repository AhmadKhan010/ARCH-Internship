# MNIST Digit Recognizer

This project includes:
- A Jupyter notebook for training and evaluating models on the MNIST dataset.
- A Gradio web app to draw digits and classify them using trained models.

## 📁 Structure

- `notebooks/mnist_training.ipynb`: Model training and evaluation
- `models/`: Saved trained models
- `app.py`: Gradio interface for digit recognition
- `requirements.txt`: Dependencies

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/AhmadKhan010/mnist-digit-recognizer.git
cd mnist-digit-recognizer
```

2. **Create and Activate Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the Models**

Open the notebook and run it:

```bash
cd notebooks
jupyter notebook mnist_training.ipynb
```

Trained models will be saved in the models/ directory.

5. Run the Gradio App
```bash
cd ..
python app.py
```

The app will launch in your browser. Draw a digit and select the model (SGD, RandomForest, or Both) to classify.


# 🖼️ Demo

Here’s how the app looks in action:

![image](https://github.com/user-attachments/assets/294856b2-0a61-486e-bf01-ade9e32991fc)
