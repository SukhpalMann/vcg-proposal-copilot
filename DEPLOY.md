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

## Public URL (live AI, free)

1. Create a free Groq key (no card): sign in at <https://console.groq.com>,
   open **API Keys**, **Create API Key**, copy it.
2. On <https://share.streamlit.io>, **Create app → Deploy a public app from
   GitHub**: this repository, branch `main`, main file `app.py`.
3. **Advanced settings**: Python 3.11, and paste into **Secrets**:

```toml
LLM_PROVIDER = "groq"
GROQ_API_KEY = "REPLACE_WITH_YOUR_GROQ_KEY"
```

4. Deploy. The sidebar should read "Live AI: gpt-oss-20b on Groq drafts each
   section (free tier)". If it says "Simulation mode", the warning states which
   key is missing; fix it under **Settings → Secrets** and reboot.

The free tier allows roughly 15 full proposals a day; a proposal takes under a
minute, including any pauses the client takes to respect the per-minute limit.
Verification is unchanged and never uses the model. Storage is ephemeral, so
saved runs may disappear after a restart. Never upload confidential client
documents to the public demo: in this mode they are sent to the provider.

## Backup

`python scripts/run_demo.py` exercises the deterministic fixture and prints the
18% supported versus 35% unsupported traceability example. It is a backup
simulation, not a substitute for showing local AI generation.
