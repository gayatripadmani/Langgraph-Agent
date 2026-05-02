# LangGraph-Agent:  LangGraph Research Assistant

GitHub repository:

```text
https://github.com/gayatripadmani/Langgraph-Agent.git
```

## Clone and Setup

1. Clone the repository:

```powershell
git clone https://github.com/gayatripadmani/Langgraph-Agent.git
cd Langgraph-Agent
```

2. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Configure `.env`

Use this format:

```env
GEMINI_API_KEY=your_gemini_key
GEMINI_CHAT_MODEL=models/gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
QDRANT_URL=https://your-qdrant-cloud-url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=research_assistant
```

5. Add your PDF files inside the `data/` folder.

6. Build the Qdrant index:

```powershell
python .\scripts\build_index.py
```

7. Start the API server:

```powershell
python .\run_api.py
```

API base URL:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

- `GET /health`
- `GET /documents`
- `GET /sample-queries`
- `POST /ask`

## cURL Tests

### Health

```powershell
curl http://127.0.0.1:8000/health
```

### Documents

```powershell
curl http://127.0.0.1:8000/documents
```

### Sample Queries

```powershell
curl http://127.0.0.1:8000/sample-queries
```

### Ask Question

```powershell
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What are the three articles in the English language?\"}"
```

### Ask Question With Thread Memory

```powershell
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Which document is longer, BCA-123 Basic English.pdf or Articles.pdf?\",\"thread_id\":\"demo-1\"}"
```

### Follow-up On Same Thread

```powershell
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"How many pages difference is there?\",\"thread_id\":\"demo-1\"}"
```

## Notes

- The project uses PDF files only from the `data/` folder
- The project uses remote Qdrant with `QDRANT_URL` and `QDRANT_API_KEY`
- Run `python .\scripts\build_index.py` before testing `/ask`
- If indexing fails with `429 RESOURCE_EXHAUSTED`, your Gemini embedding quota is exhausted
