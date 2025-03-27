# StatSolveR - An LLM-Based Python Code Execution Framework to solve Statistical Problems

A Python code execution framework powered by LLaMA 3 (70B) via OpenRouter, built with Streamlit for UI and ChromaDB for knowledge retrieval. This project allows users to input statistical queries, retrieve context from a statistics book, generate Python code using an LLM, execute it, and display results dynamically.

## 🚀 Features
✅ Query Processing – Accepts user input for statistical computations.
✅ Context Retrieval – Uses ChromaDB to fetch relevant data from JASP for Beginners.
✅ LLM Integration – Leverages LLaMA 3 (70B) via OpenRouter for generating Python code.
✅ Code Execution – Runs the LLM-generated Python code dynamically.
✅ User Interface – Built using Streamlit for an interactive experience.

## 🛠️ Tech Stack
| Component | Technology Used |
|:---------:|:---------------:|
| Frontend  | Streamlit       |
| LLM       | LLaMA 3 (70B) via OpenRouter |
| Database  | ChromaDB        |
| Backend   | Python (LangChain, NumPy, Pandas) |

## 📂 Project Structure

```console
📦 llm-python-executor/
 ┣ 📂 src/
 ┃ ┣ 📂 dependency_functions/
 ┃ ┃ ┣ functions.py        # Handles context retrieval & LLM calls
 ┃ ┃ ┣ run_code.py         # Executes Python code dynamically
 ┃ ┃ ┗ __init__.py
 ┃ ┣ 📂 ui/
 ┃ ┃ ┗ app.py              # Streamlit frontend
 ┃ ┗ __init__.py
 ┣ 📂 test/
 ┃ ┗ test.py               # Unit tests for LLM response & execution
 ┣ .env                    # API keys configuration
 ┣ requirements.txt         # Required Python libraries
 ┣ README.md               # Project documentation
 ┗ run.sh                  # Shell script to run the app
```

## 📥 Installation & Setup
### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/llm-python-executor.git
cd llm-python-executor
```

### 2️⃣ Install Dependencies
```console  
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables

Create a .env file in the project root and add the following:

```console
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
OPENROUTER_API_KEY=your_openrouter_api_key
```

### 4️⃣ Run the Streamlit App
```console
streamlit run src/ui/app.py
```

## 📊 Running & Testing the Code

### ✅ Running the Application

Once the app starts, enter a statistical query, and the system will:

- Retrieve context from JASP for Beginners via ChromaDB.
- Generate Python code using LLaMA 3 (70B).
- Execute the code dynamically and display the result.

### 🛠️ Running Tests
Run unit tests to validate execution:

```console
pytest test/test.py
```

## 🚀 Future Scope

- Agentic Framework – Implement a fully autonomous execution agent.
- Local LLM Execution – Develop an offline LLM execution system.
- Multi-Turn Conversations – Improve interaction for complex queries.

## 📚 References & Citations
### 📖 Statistics Book

**[Faulkenberry, T. (2024). JASP for Beginners. GitHub Repository](https://github.com/tomfaulkenberry/JASPbook/blob/master/README.md)**

### 🧠 Large Language Models

**[LangChain Documentation - LangChain Official](https://python.langchain.com/docs/introduction/)**

**[OpenRouter API - OpenRouter](https://openrouter.ai/)**

**[LLaMA 2 & 3 - Meta AI](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_3/)**

**[LLaMA - GGUF - The Bloke](https://huggingface.co/TheBloke)**

### 📜 Academic Citations

- **[Qwen](https://huggingface.co/Qwen)**
```bibtex
@article{qwen2.5,
      title={Qwen2.5 Technical Report}, 
      author={An Yang and Baosong Yang and Beichen Zhang and Binyuan Hui and Bo Zheng and Bowen Yu and Chengyuan Li and Dayiheng Liu and Fei Huang and Haoran Wei and Huan Lin and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Yang and Jiaxi Yang and Jingren Zhou and Junyang Lin and Kai Dang and Keming Lu and Keqin Bao and Kexin Yang and Le Yu and Mei Li and Mingfeng Xue and Pei Zhang and Qin Zhu and Rui Men and Runji Lin and Tianhao Li and Tianyi Tang and Tingyu Xia and Xingzhang Ren and Xuancheng Ren and Yang Fan and Yang Su and Yichang Zhang and Yu Wan and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zihan Qiu},
      journal={arXiv preprint arXiv:2412.15115},
      year={2024}
}
```

- **[PolyCoder](https://huggingface.co/NinedayWang)**
```bibtex
@inproceedings{
  xu2022polycoder,
  title={A Systematic Evaluation of Large Language Models of Code},
  author={Frank F. Xu and Uri Alon and Graham Neubig and Vincent Josua Hellendoorn},
  booktitle={Deep Learning for Code Workshop},
  year={2022},
  url={https://openreview.net/forum?id=SLcEnoObJZq}
}
```

## 👨‍💻 Author & Contributions

**🚀 Kailash Velumani – Data Scientist @ Accenture**

**📢 Contributions are welcome! Open an issue or pull request for improvements. 🎯**

📧 Contact: 
 - [LinkedIn - Kailash V](https://www.linkedin.com/in/kailash-v/)
 - [Email](mail to: veluvkl@outlook.com): veluvkl@outlook.com 