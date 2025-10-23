Railway deployment guide (uv dependency manager)

1) Prepare environment
- Copy `.env.example` to `.env` and fill in your secrets.

2) Deploy to Railway
- On Railway, create a new project and choose GitHub repo deployment or connect your repository.
- In Railway's settings for the project, select "uv" as the dependency manager and upload the `uv.json` file from the repo root (Railway will detect it automatically if present).

3) Environment variables
Make sure the variables from `.env` are configured in Railway: `SUPABASE_URL`, `SUPABASE_KEY`, `REDIS_HOST`, `REDIS_PASSWORD`, `RELOAD_SECRET_TOKEN`, `QUESTIONS_PATH`, and `PORT`.

4) Start command
The `Procfile` and `uv.json` specify the start command:

web: uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1

5) Notes and tips
- This project uses FastAPI and Uvicorn. We recommend 1 worker on Railway for lightweight apps; increase workers only if you provision enough CPU.
- If you use GPUs or heavy PyTorch work, configure a machine with more resources and consider caching the built engine externally.
- Ensure the `questions.json` file is available; set `QUESTIONS_PATH` to the file path or store the file in the repo.

6) Optional: Health checks
- Add an HTTP health check to `/stats` or `/question/<some-test-session>` in Railway so it can confirm the app is healthy.

7) Local testing
- To run locally using uv (similar to Railway):

  # create virtual env and install
  python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
  # run app
  $env:PORT=8000; uvicorn main:app --host 0.0.0.0 --port $env:PORT --reload

