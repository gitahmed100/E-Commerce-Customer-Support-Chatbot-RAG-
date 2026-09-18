# E-commerce RAG Support Chatbot — Full Step-by-Step Guide

This is a complete walkthrough, written for someone who has never built a RAG
system before. Follow it in order. Every code block is meant to be copy-pasted.

## 0. What you're building, in plain terms

Every customer message goes through 4 small models, one after another:

```
customer message
   -> [1] Language Detection      "what language is this?"
   -> [2] Sentiment Classifier    "are they upset?"
   -> [3] Intent Classifier       "what do they actually want?"
   -> [4] RAG (retrieve + LLM)    "look up the real answer and write a reply"
   -> final reply
```

Modules 1–3 are small classifiers you train yourself on labeled data.
Module 4 (RAG) doesn't get "trained" — instead you build a searchable database
of past support answers, and at chat-time you look up the most relevant ones
and hand them to an LLM to write the final reply. That's what "Retrieval-
Augmented Generation" means: the LLM's generation is *augmented* by retrieved
real data, instead of relying only on what it memorized during its own training.

## 1. Where to work

You need two different environments for two different steps:

| Step | Where | Why |
|---|---|---|
| Notebooks 1, 3, 4 (language, intent, RAG) | **Local Jupyter** or Google Colab — CPU is fine | Fast, lightweight models |
| Notebook 2 (sentiment/Transformer fine-tuning) | **Google Colab** (free GPU) | Fine-tuning DistilBERT is much faster on GPU |
| Deployment (`app/`) | **Local machine** (or wherever you'll demo it) | Needs to run continuously and answer live requests |

If you don't want to juggle two environments, you can run everything in Colab
(just slower for Notebook 2 on CPU) and then download the saved `models/`
folder to your machine afterward for deployment.

## 2. One-time setup on your local machine

Open a terminal and run:

```bash
# 1. Create a project folder and move into it (skip if you already unzipped the delivered project)
cd ecommerce-rag-chatbot

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt
```

This installs everything all 4 notebooks and the deployment app need
(scikit-learn, transformers, sentence-transformers, faiss, groq, fastapi, etc).

## 3. Get your free Groq API key (needed for Module 4 only)

1. Go to https://console.groq.com and sign up — it's free.
2. Click **API Keys** in the sidebar → **Create API Key**.
3. Copy the key (starts with `gsk_...`).
4. Set it as an environment variable so no notebook or script has your key
   hardcoded in it:
   ```bash
   export GROQ_API_KEY=gsk_your_key_here      # on Windows (PowerShell): $env:GROQ_API_KEY="gsk_..."
   ```
   Run this in the same terminal you'll launch Jupyter / uvicorn from. If you
   restart your terminal, you'll need to set it again (or add it to your
   shell profile, e.g. `~/.bashrc` or `~/.zshrc`).

You do **not** need a Qdrant account — this project uses a local FAISS vector
store by default, which needs no signup at all (Notebook 4 includes optional,
commented-out code showing how to swap in cloud Qdrant if you want that instead).

## 4. Run the 4 notebooks, in this exact order

Launch Jupyter from inside the project folder:

```bash
jupyter notebook
```

This opens a browser tab. Open `notebooks/` and run each notebook **top to
bottom** (Cell → Run All, or click through cells with Shift+Enter), in this order:

1. **`01_language_detection.ipynb`**
   Downloads the language dataset, trains a TF-IDF + Logistic Regression
   classifier, evaluates it, and saves it to `models/language_detector.joblib`.
   Takes under a minute.

2. **`03_intent_classifier.ipynb`**
   Downloads the Bitext support dataset, maps its 27 intents into 7 routing
   buckets, trains a TF-IDF + Linear SVM classifier, saves it to
   `models/intent_classifier.joblib`. Takes under a minute.
   **Important:** this notebook prints any intent names that didn't map
   cleanly into a bucket — read that output and extend the mapping dictionary
   if your copy of the dataset uses slightly different intent names.

3. **`04_rag_pipeline.ipynb`**
   Builds sentence embeddings for the whole support knowledge base, stores
   them in a local FAISS index (`models/rag_store/`), and shows you how to
   call the Groq LLM with retrieved context. Needs your `GROQ_API_KEY` from
   Step 3 above. Takes a few minutes (mostly the embedding step).

4. **`02_sentiment_classifier.ipynb`**
   Fine-tunes a small Transformer (DistilBERT) to classify messages as
   negative/neutral/positive. **Run this one in Google Colab with a GPU**
   (Runtime → Change runtime type → T4 GPU) — on CPU it will still work but
   can take 15–20+ minutes. When done, it saves to `models/sentiment_model/`.
   If you ran it in Colab, download that whole folder afterward and place it
   inside your local project's `models/sentiment_model/`.

After all 4 notebooks finish, your `models/` folder should look like this:

```
models/
├── language_detector.joblib
├── intent_classifier.joblib
├── sentiment_model/
│   ├── config.json
│   ├── model.safetensors (or pytorch_model.bin)
│   ├── tokenizer.json
│   └── ...
└── rag_store/
    ├── kb.index
    └── kb_metadata.pkl
```

If any of those files/folders are missing, re-run the corresponding notebook —
the deployment app will fail to start without all four.

## 5. Run the deployed chatbot API

Make sure `GROQ_API_KEY` is still set in your terminal (Step 3), then from the
project root run:

```bash
uvicorn app.main:app --reload --port 8000
```

You should see log lines like `Loading language detector...` down to
`All models loaded. Ready.`, then Uvicorn will say it's running on
`http://127.0.0.1:8000`.

### Try it in the browser
Open **http://127.0.0.1:8000/docs** — this is an interactive API tester
(Swagger UI) generated automatically by FastAPI. Click on `POST /chat`,
"Try it out", type a message like `"Where is my order?"` into the JSON body,
and click Execute to see the full response.

### Try it from the terminal
In a **second** terminal window (leave the server running in the first):

```bash
curl -X POST http://127.0.0.1:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Where is my order? Its been 5 days"}'
```

### Try it with the included test script
```bash
python test_api.py
```
This sends 5 example messages (a greeting, an order question, a complaint, an
account question, and an out-of-scope question) and prints the full response
for each — a good way to sanity-check that routing behaves correctly for each
case before your demo.

## 6. How the routing logic works (for your assessment Q&A)

This lives in `app/pipeline.py`, function `handle_message()`:

- **`greeting`** → answered directly with a canned reply, no RAG call (fast,
  and avoids wasting an LLM call on small talk).
- **`complaint`** → always escalated to a human, regardless of sentiment
  score, with an immediate apology sent back so the customer isn't left in
  silence. *(Documented design choice: complaints are inherently about broken
  trust — a generated answer risks sounding dismissive, so a human handles
  these ones, not the bot.)*
- **`order_status` / `order_management` / `billing_and_refunds` /
  `account_management`** → goes through the full RAG pipeline: retrieve top-3
  similar past Q&A pairs, generate a grounded answer with Groq's LLM. If the
  detected sentiment is `negative`, an extra apology sentence is prepended
  *in code* (not just relying on the LLM prompt instruction) as a guaranteed
  tone safeguard.
- **`out_of_scope`** → a canned "can you rephrase" message, no RAG or LLM
  call (nothing relevant would be retrieved anyway).

## 7. Deliverables checklist (matches what your assignment asks for)

- ✅ Four module-specific notebooks → `notebooks/01_...` through `04_...`
- ✅ Deployment scripts → `app/main.py` (FastAPI server) + `app/pipeline.py`
  (the shared inference logic all 4 modules feed into)
- ✅ Documentation → this `README.md`, plus inline markdown cells inside every
  notebook explaining each step and design choice as you go

## 8. Things you'll likely be asked about (so think through your answers)

- Why TF-IDF + character n-grams for language ID, but word n-grams for
  intent? (Character patterns are language-specific and work on any length
  of text; word patterns capture the *meaning* of a full sentence, which is
  what intent depends on.)
- Why collapse 6 emotions into 3 buckets? (The routing system only needs to
  know "should this get a softer tone," not which specific emotion — fewer
  classes also means better accuracy on a small, out-of-domain Twitter
  dataset.)
- Why embed the *questions* in the knowledge base rather than the *answers*?
  (A new customer question is phrased like a question, so it matches other
  questions better than it matches answer-style text.)
- What happens if retrieval finds nothing relevant? (The system prompt
  explicitly tells the LLM to say so honestly and suggest escalation, rather
  than inventing an answer — this is the core anti-hallucination guardrail
  in a RAG system.)
- Why complaints bypass RAG entirely? (Documented design choice above —
  be ready to defend it, or to argue the alternative: RAG-answer complaints
  too, but flag them for follow-up.)







To run the project commands in 2 differnt terminals: 

Terminal 1:
$env:GROQ_API_KEY="YOUR_NEW_GROQ_API_KEY"
C:\Users\Smart\AppData\Local\Python\pythoncore-3.14-64\python.exe -m uvicorn app.main:app --reload --port 8000

Terminal 2:
C:\Users\Smart\AppData\Local\Python\pythoncore-3.14-64\python.exe -m streamlit run streamlit_app.py