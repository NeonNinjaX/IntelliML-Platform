# IntelliML Platform 🧠🎙️📊

**IntelliML** is a next-generation **Voice-Controlled AutoML Platform**. It goes beyond traditional data tools by enabling users to perform complex machine learning tasks, data analysis, and model training using natural language commands.

Built with a modular architecture, it combines a high-performance **FastAPI** backend, a stunning **Next.js** frontend, and a dedicated **ML Engine** for robust data processing and explainable AI.

---

## ✨ Key Features

### 1. **Voice-Controlled Analytics** 🎙️
*   **Natural Language Interface**: Talk to your data using Groq-powered NLU.
*   **Voice Commands**: "Train a model", "Explain this prediction", "Clean the data".
*   **Text-to-Speech**: The platform talks back, explaining insights and actions.

### 2. **Automated Machine Learning (AutoML)** 🤖
*   **Model Training**: Support for **Random Forest**, **XGBoost**, **LightGBM**, and Linear models.
*   **Auto-Tuning**: Automatically selects the best hyperparameters.
*   **Model Comparison**: Visualize performance metrics (ROAUC, Confusion Matrix, F1-Score) side-by-side.

### 3. **Explainable AI (XAI)** 💡
*   **SHAP Integration**: Understand *why* a model made a specific prediction.
*   **Feature Importance**: See which variables drive your model's decisions.
*   **Global & Local Explanations**: Analyze model behavior at both the dataset and individual record levels.

### 4. **Smart Data Preparation** 🛠️
*   **AI Health Report**: Automated detection of missing values, outliers, and encoding needs.
*   **Interactive Cleaning Station**:
    *   **Imputation**: Fill missing values (Mean/Median/Mode/Zero).
    *   **Transformation**: Scale (Standard/MinMax) and Encode (One-Hot/Label) data.
    *   **Outlier Management**: Detect and remove anomalies using IQR.

### 5. **Modern Visualization & UI** 🎨
*   **Interactive Dashboards**: Tremor and Recharts integrations for stunning data viz.
*   **Dark Mode**: A premium "Obsidian & Electric Blue" aesthetic.
*   **Real-time Updates**: WebSocket integration for live training progress and voice feedback.

---

## 🏗️ Architecture & Tech Stack

The project follows a modular 3-tier architecture:

### **Frontend** (`/frontend`)
*   **Framework**: [Next.js 14](https://nextjs.org/) (React)
*   **Styling**: [Tailwind CSS](https://tailwindcss.com/)
*   **State**: React Hooks & Context
*   **Visualization**: Recharts, Tremor, Framer Motion

### **Backend** (`/backend`)
*   **API**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
*   **ML Engine**: [Scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/)
*   **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **NLU & Voice**: Groq API, SoundDevice, TTS Services

### **ML Engine** (`/ml_engine`)
A standalone module containing core logic:
*   `engines/`: Algorithm implementations.
*   `mcp_servers/`: Model Context Protocol integration.
*   `explanation_service.py`: SHAP/LIME logic.

---

## 🚀 Getting Started

### Prerequisites
*   **Node.js** (v18+)
*   **Python** (v3.9+)
*   **Groq API Key** (for Voice/NLU features)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/IntelliML-Platform.git
cd IntelliML-Platform
```

### 2. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up Environment Variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run Server
uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Run App
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to start using IntelliML.

---

## 📂 Project Structure

```
IntelliML-Platform/
├── backend/
│   ├── app/
│   │   ├── api/            # Routes (data, voice, ml)
│   │   ├── services/       # NLU, Voice, Explanation services
│   │   └── main.py         # Entry point
│   └── requirements.txt
│
├── frontend/
│   ├── app/                # Next.js Pages
│   ├── components/         # UI & Visualizations
│   └── public/             # Assets
│
└── ml_engine/              # Core ML Logic
    ├── engines/            # Model implementations
    └── mcp_servers/        # MCP Servers
```

---

## 🔌 API Documentation
Full interactive API documentation provided by Swagger UI:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🛡️ License
This project is licensed under the **MIT License**.
