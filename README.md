# 🌱 Smart Biogas Optimization System

## Short Description
An AI-powered system that predicts biogas production using an Artificial Neural Network (ANN), optimizes waste composition using a Genetic Algorithm (GA), and provides intelligent recommendations through an AI advisor. The system helps improve efficiency and supports sustainable energy solutions.

---

## Features
- 🔥 Biogas prediction using ANN model  
- ♻️ Waste mix optimization using Genetic Algorithm  
- 🤖 AI-based advisory system for actionable insights  
- 📊 Interactive data visualizations (scatter plot, heatmap, bar chart)  
- 🎛️ User-friendly dashboard built with Streamlit  

---

## Technologies Used
- Python  
- Streamlit  
- TensorFlow / Keras  
- Scikit-learn  
- Pandas & NumPy  
- Plotly  
- Groq API (LLM integration)  

---

## Project Structure
biogas-optimization-ai/
│
├── app.py # Main Streamlit application
├── optimizer.py # Genetic Algorithm for optimization
├── groq_advisor.py # AI advisor using Groq API
├── train_model.py # Model training script
├── dataset_waste.csv # Dataset
├── model.h5 # Trained ANN model
├── scaler.pkl # StandardScaler
├── README.md # Project documentation
├── .gitignore # Ignored files

---

## How to Run

1. Clone the repository:
   git clone https://github.com/nvaishnavi029-lang/smart-biogas-optimizer.git
   cd smart-biogas-optimizer

3. Install dependencies:
   pip install -r requirements.txt

4. Set API Key (for AI advisor):
   set GROQ_API_KEY=your_api_key # Windows
   export GROQ_API_KEY=your_api_key # Mac/Linux

5.  Run the application:
    streamlit run app.py

---

## Author
**Vaishnavi N**  
AI & ML Enthusiast  

---

