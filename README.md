# IntelliML Platform 🧠📊🤖

**IntelliML** is an AI-powered analytics platform that revolutionizes data science workflows. Featuring an intelligent AI assistant, automated machine learning capabilities, and a stunning warm retro-themed interface, it enables both beginners and experts to perform sophisticated data analysis through natural language.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-intelli--ml--platform.vercel.app-orange?style=for-the-badge&logo=vercel)](https://intelli-ml-platform.vercel.app)
[![GitHub](https://img.shields.io/badge/GitHub-NeonNinjaX-black?style=for-the-badge&logo=github)](https://github.com/NeonNinjaX)


---

## ✨ Key Features

### 🤖 AI-Powered Data Assistant
- **Natural Language Queries** — Ask questions about your data in plain English
- **Code Generation** — Automatically generates Python code for data analysis tasks
- **Interactive Visualizations** — Creates matplotlib visualizations on-demand
- **Collapsible Code Blocks** — Clean interface with code hidden by default
- **Copy-to-Clipboard** — Easy code sharing with instant feedback

### 🧹 Intelligent Data Cleaning
- **Missing Value Detection** — Automatic identification and handling
- **Multiple Imputation Methods** — Mean, Median, Mode, Zero, Forward Fill, Backward Fill
- **Outlier Detection** — IQR-based anomaly detection with visualization
- **Column Management** — Easy deletion of unwanted features

### 📊 Exploratory Data Analysis (EDA)
- **Statistical Summaries** — Comprehensive dataset statistics
- **Distribution Analysis** — Histograms and density plots
- **Correlation Heatmaps** — Visualize feature relationships
- **Missing Data Visualization** — Identify data quality issues at a glance

### ⚙️ Feature Engineering
- **Data Scaling** — StandardScaler and MinMaxScaler support
- **Encoding** — One-Hot and Label encoding for categorical variables
- **Custom Transformations** — Build advanced feature pipelines
- **Real-time Preview** — See transformations before applying

### 🎯 Automated Machine Learning (AutoML)
- **Multiple Algorithms** — Random Forest, XGBoost, LightGBM, Logistic Regression
- **Auto-Tuning** — Intelligent hyperparameter optimization
- **Model Comparison** — Side-by-side performance metrics
- **Explainable AI** — SHAP integration for model interpretability

### 🎨 Modern Design
- **Warm Retro Theme** — Elegant amber, cream, and burgundy color palette
- **Responsive Layout** — Works seamlessly on all screen sizes
- **Smooth Animations** — Delightful user experience with subtle motion
- **Accessibility** — High contrast and readable typography

---

## 🏗️ Architecture & Tech Stack

### Frontend (`/frontend`)
| Technology | Purpose |
|---|---|
| [Next.js 14](https://nextjs.org/) + TypeScript | Core framework |
| [Tailwind CSS](https://tailwindcss.com/) | Styling |
| shadcn/ui | UI component primitives |
| Recharts | Data visualization |
| React Hooks & Context API | State management |

### Backend (`/backend`)
| Technology | Purpose |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | API framework |
| Scikit-learn, XGBoost, LightGBM | ML libraries |
| Pandas, NumPy | Data processing |
| Groq API (Llama 3.3 70B) | AI integration |
| Matplotlib + base64 | Visualization rendering |

---

## 🚀 Getting Started

### Prerequisites
- **Node.js** v18+
- **Python** v3.9+
- **Groq API Key** — Get yours at [console.groq.com](https://console.groq.com)

### 1. Clone the Repository
```bash
git clone https://github.com/NeonNinjaX/IntelliML-Platform.git
cd IntelliML-Platform
```

### 2. Backend Setup
```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GROQ_API_KEY=your_api_key_here" > .env

# Start the backend server
python run.py
```

Backend runs at `http://localhost:8000`

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to access IntelliML.

---

## 📂 Project Structure

```
IntelliML-Platform/
├── backend/
│   ├── app/
│   │   ├── api/                        # API route handlers
│   │   │   ├── chat.py                 # AI Assistant endpoints
│   │   │   ├── data.py                 # Data processing endpoints
│   │   │   └── ml.py                   # ML training endpoints
│   │   ├── services/                   # Business logic
│   │   │   ├── data_chat_service.py    # AI chat with code execution
│   │   │   ├── groq_client.py          # Groq API integration
│   │   │   └── ml_service.py           # Model training service
│   │   ├── config.py                   # Application configuration
│   │   └── main.py                     # FastAPI entry point
│   ├── requirements.txt                # Python dependencies
│   └── run.py                          # Server launcher
│
├── frontend/
│   ├── app/                            # Next.js App Router
│   │   ├── page.tsx                    # Main dashboard
│   │   ├── layout.tsx                  # Root layout
│   │   └── globals.css                 # Global styles + animations
│   ├── components/
│   │   ├── landing/                    # Landing page components
│   │   ├── chat/                       # AI Assistant UI
│   │   ├── data/                       # Data cleaning & EDA
│   │   └── ml/                         # ML training components
│   ├── lib/
│   │   └── api.ts                      # API client utilities
│   └── public/                         # Static assets
│
└── README.md
```

---

## 🔌 API Documentation

Full interactive API docs powered by Swagger UI:

👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

### Key Endpoints

#### AI Assistant
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat/message` | Send message to AI assistant |
| `GET` | `/api/chat/suggestions` | Get visualization suggestions |
| `POST` | `/api/chat/clear` | Clear chat history |

#### Data Processing
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/data/upload` | Upload CSV dataset |
| `GET` | `/api/data/health` | Get data quality report |
| `POST` | `/api/data/clean` | Apply data cleaning operations |
| `POST` | `/api/data/transform` | Feature engineering transformations |

#### Machine Learning
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/ml/train` | Train ML model |
| `GET` | `/api/ml/models` | List available models |
| `POST` | `/api/ml/explain` | Generate SHAP explanations |

---

## 💡 Usage Examples

### Using the AI Assistant
1. Upload your CSV dataset
2. Navigate to the **AI Assistant** tab
3. Ask questions like:
   - *"Show me a correlation heatmap"*
   - *"Create a histogram of all numeric columns"*
   - *"What are the most important features?"*
4. The AI generates code, executes it, and displays visualizations

### Training a Model
1. Clean your data in the **Data Cleaning** tab
2. Engineer features in **Feature Engineering**
3. Go to **Train** and select your target variable, algorithm, and hyperparameters
4. View results with metrics, charts, and SHAP explanations

---

## 🎯 Roadmap

- [ ] Model Deployment — One-click model export and API generation
- [ ] Advanced Visualizations — Plotly integration for interactive charts
- [ ] Team Collaboration — Share datasets and models with teammates
- [ ] AutoML Pipelines — Save and reuse complete ML workflows
- [ ] Custom Models — Upload and integrate your own models
- [ ] Real-time Predictions — Live inference on streaming data

---

## 🐛 Known Issues

- **Download Button** — Some browsers may block automatic downloads from localhost. Check your browser's download permissions if visualizations don't download.
- **Groq API Rate Limits** — Free tier has daily token limits. Upgrade to Pro for higher limits.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 🙏 Acknowledgments

- [Groq](https://groq.com) for lightning-fast LLM inference
- [Next.js](https://nextjs.org) team for the amazing React framework
- [FastAPI](https://fastapi.tiangolo.com) for the elegant Python backend
- [Tailwind CSS](https://tailwindcss.com) for the utility-first CSS framework
- [shadcn/ui](https://ui.shadcn.com) for beautiful component primitives

---

## 📧 Contact

- **GitHub**: [@NeonNinjaX](https://github.com/NeonNinjaX) — mishrarahul2898@gmail.com
- **GitHub**: [@Theani7](https://github.com/Theani7)

---

<div align="center">

**Built with ❤️ using Next.js, FastAPI, and Groq AI**

[⭐ Star this repo](https://github.com/NeonNinjaX/IntelliML-Platform) if you find it helpful!

</div>
