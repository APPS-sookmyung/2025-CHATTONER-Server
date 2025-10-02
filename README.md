# ChatToner  

**A Literacy-Based Personalized Tone Conversion System**
ChatToner is an intelligent educational tone conversion system that finds the optimal tone by considering the user's literacy level and the specific context of the conversation.

Deployment URL: https://client-184664486594.asia-northeast3.run.app

---

## Project Overview

Going beyond simple tone conversion, ChatToner helps facilitate the most effective communication by diagnosing a user's **literacy level** and reflecting the specific **context of the conversation, including its purpose, situation, and audience**. The system identifies a user's language skills on a session basis and provides level-specific feedback and sophisticated conversion results, ultimately aiming to contribute to the improvement of literacy.

---

## Key Features

### Literacy Level Diagnosis & Personalized Feedback
We provide a user-optimized experience with a session-based literacy diagnosis and dynamic feedback system.
-   **Initial Literacy Diagnosis**: Identifies the user's basic literacy level (Beginner/Intermediate/Advanced) through a simple text comprehension test.
-   **Level-Specific Custom Feedback**: Provides differentiated feedback based on the diagnosed level:
    -   **Beginner**: Simple, intuitive improvement suggestions focusing on basic vocabulary.
    -   **Intermediate**: Concrete feedback focusing on grammatical accuracy and expressive power.
    -   **Advanced**: Professional feedback on advanced vocabulary usage, stylistic completeness, and logical structure.
-   **Dynamic Level Adjustment**: Automatically adjusts the feedback level in real-time by detecting improvements in the user's language skills during service use.

### Sophisticated Tone Conversion Based on Context
Users can directly set detailed situational information to generate sophisticated conversion results tailored to the actual situation, rather than mechanical conversions.
-   **Detailed Context Settings**: Allows users to specify the following through dropdown menus:
    -   **Audience**: Elementary students, university students, teachers, parents, etc.
    -   **Communication Purpose**: Class explanations, assignment instructions, evaluation feedback, etc.
    -   **Difficulty & Situation**: Beginner/Intermediate/Advanced, regular class/individual tutoring, etc.
-   **Specialized Educational Tone Conversion**: Provides conversions optimized for language education and literacy improvement.
    -   **Intelligent Vocabulary Explanation**: Automatically identifies difficult vocabulary and provides explanations with examples suited to the learner's level.
    -   **Phased Sentence Complexity Control**: Optimizes sentence length and use of conjunctions to match the learner's level.
    -   **Enhanced Context-Based Explanations**: Improves comprehension by using analogies and examples connected to real life.

### Multi-Dimensional Quality Evaluation & Analysis
We go beyond simple style comparisons to comprehensively analyze the quality of conversion results and suggest improvements through a multi-dimensional evaluation system.
-   **Three Core Evaluation Axes**:
    1.  **Learner-Fit Evaluation**: Assesses comprehension level and vocabulary suitability for the target learner.
    2.  **Context-Fit Evaluation**: Checks logical flow and delivery in light of the educational purpose.
    3.  **Style-Fit Evaluation**: Verifies the naturalness of the sentence and compliance with stylistic constraints (e.g., negative prompts).
-   **Enhanced Embedding and RAG-Based Scoring**: Combines an embedding model covering the full scope of literacy education with a RAG-based analysis logic to provide objective, scored learning metrics.

---

## Tech Stack

| Category | Stack/Library | Purpose |
| :--- | :--- | :--- |
| **Frontend** | React.js, TypeScript, Zustand | UI, Type Safety, State Management |
| **Backend** | FastAPI | Unified API Server, ML Model Serving |
| **ML** | LoRA, KoGPT, KoAlpaca, HuggingFace Transformers | Style Conversion, Text Generation, RAG |
| **Database** | PostgreSQL, FAISS, Redis | Data Storage, Vector Search, Caching |

---

## Project Structure
chattoner/
├── client/
│ └── ... (React Frontend)
├── python_backend/
│ ├── app/
│ ├── ml/
│ └── requirements.txt
├── docker-compose.yml
└── README.md
*The server architecture has been consolidated into a single FastAPI server to improve maintainability.*

---

## Installation & Setup

1.  **Clone the project**
    ```
    git clone [https://github.com/your-username/chat-toner.git](https://github.com/your-username/chat-toner.git)
    cd chat-toner
    ```
2.  **Set up the environment**
    ```
    cd client && npm install # Client
    cd ../python_backend && pip install -r requirements.txt
    ```
3.  **DB & Environment Variables**
    ```
    createdb chattoner
    cp .env.example .env
    # Modify DB connection info in .env
    ```
4.  **Run the application**
    ```
    docker-compose up --build
    ```

---

## Evaluation Method
ChatToner comprehensively evaluates conversion quality based on three key axes: **Learner-Fit Evaluation**, **Context-Fit Evaluation**, and **Style-Fit Evaluation**. It derives objective scores through enhanced embeddings and RAG, providing users with practical improvement metrics.

---

## Contributing

1.  Fork this repo
2.  Create a feature branch (`git checkout -b feature/NewFeature`)
3.  Commit your changes (`git commit -m 'Add ...'`)
4.  Push to the branch (`git push origin feature/NewFeature`)
5.  Submit a Pull Request

---

## Team

| Name | Role | Responsibilities | Contact |
| :--- | :--- | :--- | :--- |
| **Yoon Jiwon** | PM | Project Planning, Management | geenieeyoon@gmail.com |
| **Kwon Yujin** | Development | Full-Stack, ML Model Support | thinz0083@daum.net |
| **Kim Jimin** | Development | Fine-tuning, RAG Implementation | onlypotato637@gmail.com |
| **Jeong Jieun** | Development | Model Development, Fine-tuning | jje49jieun@gmail.com |
| **Ha Jimin** | Development | Frontend Design/Implementation, API Integration | tracygkwlals@gmail.com |
