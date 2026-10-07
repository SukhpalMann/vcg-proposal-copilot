# Demonstration setup

## Full offline AI demonstration (recommended for the course)

Run the application and Ollama on the same laptop. Install dependencies with
`pip install -r requirements.txt`, install Ollama, and download the model once
while connected to the internet:

```bash
ollama pull gemma3
python scripts/seed_corpus.py
LLM_PROVIDER=ollama LLM_MODEL=gemma3:latest streamlit run app.py
```

After the model is downloaded, the app calls only `127.0.0.1` for generation.
Keep **Include external market context** disabled; the UI disables it in local
mode. The CRM, HR, rate-card, and time/billing records are fictional files in
`data/sample_systems/`. There are no live enterprise-system connections.

Test the entire path on the presentation laptop. Local extraction can be slow,
especially on long PDFs. Prepare a full run in advance and time a short live
section demonstration. Clearly label prepared and live outputs. Show the active
provider and token/latency log in the Execution tab.

## Public URL (live AI with a hosted model)

On Streamlit Community Cloud, choose this repository and `app.py` as the entry
point. Then open the app's **Settings → Secrets** and paste:

```toml
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-haiku-5-5"
ANTHROPIC_API_KEY = "REPLACE_WITH_YOUR_KEY"
```

(The same lines are in `.streamlit/secrets.toml.example`.) Reboot the app. The
sidebar should read "Live AI: claude-haiku-5-5 drafts each section". If it still
says "Simulation mode", the key was not picked up; the warning states why.

A full proposal drafts in seconds and costs well under a rupee in tokens; the
Execution tab shows the measured figure. Verification is unchanged and never
uses the model. Storage is ephemeral, so saved runs may disappear after a
restart. Never upload confidential client documents to the public demo: in this
mode they are sent to Anthropic's API.

## Backup

`python scripts/run_demo.py` exercises the deterministic fixture and prints the
18% supported versus 35% unsupported traceability example. It is a backup
simulation, not a substitute for showing local AI generation.
