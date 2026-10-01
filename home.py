"""
Home.py - Lab 2 Evidence Exploration app.

For each critical risk, an AI agent searches the ChromaDB evidence database
with a search tool (function calling), looping until it has enough evidence.
The findings are then combined into a due diligence report for the Board.

Run with:  streamlit run Home.py
(Build the database first with:  python ingest.py)
"""

# Streamlit Cloud needs a newer sqlite3; this is a no-op if pysqlite3 is absent
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import json
import os

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------- Settings ----------
MODEL = "gpt-4.1"              # chat model; change to the model your course uses
EMBEDDING_MODEL = "text-embedding-3-large"  # must match ingest.py
DB_PATH = "chroma_db"
COLLECTION_NAME = "canvassian"
N_RESULTS = 6                  # chunks returned per search
MAX_ROUNDS = 6                 # safety valve for the agent loop

client = OpenAI()  # reads OPENAI_API_KEY from the environment


@st.cache_resource
def get_collection():
    """Open the database once and reuse it across reruns."""
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=EMBEDDING_MODEL,
    )
    db = chromadb.PersistentClient(path=DB_PATH)
    return db.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)


# ---------- The tool the agent can use ----------
def search_documents(query: str, doc_type: str = None) -> str:
    """Search the evidence database and return the matching chunks as text."""
    where = {"doc_type": doc_type} if doc_type else None
    results = get_collection().query(
        query_texts=[query], n_results=N_RESULTS, where=where
    )
    blocks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        blocks.append(f"SOURCE: {meta['source']}\n{doc}")
    return "\n\n---\n\n".join(blocks) if blocks else "No results."


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Semantic search over the Canvassian due diligence documents "
                "(emails, contracts, board papers). Returns the most relevant "
                "text chunks with their source file names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for, in natural language.",
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": ["emails", "contracts", "board_papers"],
                        "description": "Optional filter to one document type.",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

# ---------- Instructions for the agent ----------
AGENT_PROMPT = """You are a senior M&A lawyer conducting urgent legal due diligence.
You act for the PURCHASER, which is considering acquiring Canvassian Pty Ltd,
a cybersecurity software company. The evidence is a database of Canvassian's
emails, contracts and board papers, which you search with the search_documents tool.

How to work:
- Search several times with different wordings and angles. Use the doc_type
  filter when it helps (e.g. contracts for contractual terms).
- Many emails are irrelevant (spam, routine updates). Ignore them.
- Distinguish confirmed facts from rumours, opinions and speculation.
- If you cannot find evidence on a point, say so expressly. Do not invent facts.
- Cite the exact SOURCE path for every finding, e.g.
  contracts/Canvassian_Bravocat_Contract_Agreement_2023_3.txt, and briefly quote
  the key words that support it.
- Label each finding CONFIRMED (stated in a document), REPORTED (rumour,
  second-hand or opinion) or NOT FOUND.

When you have enough evidence, stop searching and reply with your findings:
a short summary, the key evidence with sources, and your assessment of the risk
(High / Medium / Low) and its potential impact on the purchase price."""

TOPICS = {
    "Key person risk: Jane Wu": (
        "Investigate whether the founder, Jane Wu, is likely to remain involved and "
        "motivated to lead Canvassian after the acquisition. Look for any signs of "
        "planned departure, dissatisfaction, disputes, or relevant terms in her "
        "employment or retention arrangements. State who said what, and when."
    ),
    "PayWise financial position": (
        "PayWise Pty Ltd is Canvassian's largest client (about 20% of revenue). "
        "Investigate the rumours that PayWise is in financial difficulty: late or "
        "missed payments, disputes, renegotiation requests, or other warning signs. "
        "Distinguish documented facts from rumour. Assess the likely revenue impact."
    ),
}

# One separate investigation per major client, so none is skipped
MAJOR_CLIENTS = ["PayWise", "Alphabear", "Bravocat", "Charlemont", "Deltaforce", "Echona"]
for name in MAJOR_CLIENTS:
    TOPICS[f"Change of control: {name}"] = (
        f"Review Canvassian's contract(s) with {name} only. Search the contracts "
        f"(doc_type 'contracts') and include '{name}' in your queries. Report whether "
        "there is a change of control clause, or an assignment or termination clause, "
        "that could be triggered by a sale of Canvassian's shares. If so, state the "
        "exact trigger, the counterparty's rights (consent, termination, "
        "renegotiation), any notice period and the consequences. If no such clause "
        "can be located, say 'Not located' and list the documents you reviewed."
    )

TOPICS["Other significant risks"] = (
    "Look for any other significant risks to the purchaser not covered by key "
    "person, PayWise or change of control issues, for example litigation, "
    "regulatory problems, data breaches, IP ownership, employee issues or "
    "financial irregularities. For litigation, state the parties, the claim, "
    "its status and any amount claimed."
)

REPORT_PROMPT = """You are a senior M&A lawyer. Using ONLY the findings below,
write a due diligence report for the purchaser's Board, which must decide this
afternoon whether to proceed with the acquisition of Canvassian Pty Ltd.

Structure:
1. Executive summary with an overall recommendation
2. One section per critical risk (key person, PayWise, change of control, other).
   For each: findings, evidential basis (Confirmed / Reported / Not found), risk
   rating and price impact. Quantify revenue at risk where the findings allow
   (PayWise is about 20% of revenue; the six major clients together about 60%).
3. For change of control, a table with one row for EACH of PayWise, Alphabear,
   Bravocat, Charlemont, Deltaforce and Echona: clause located (Yes / No /
   Not located), trigger, counterparty rights, source.
4. Recommended protections, each tied to a specific risk above (e.g. counterparty
   consents or waivers as conditions precedent, a new service or retention
   agreement with Jane Wu, specific indemnities, price adjustment or escrow).
5. Limitations of this review (automated, time-limited, not a full due diligence)

Rules:
- Cite exact source file paths in backticks, e.g.
  `contracts/Canvassian_Bravocat_Contract_Agreement_2023_3.txt`.
  Never cite "various" documents and never use hyperlinks.
- Do not add facts that are not in the findings.
- "Not located" means this review did not find a clause, not that none exists.
  Say so in the report."""


# ---------- The agent loop ----------
def investigate(task: str, log) -> str:
    """Let the agent search repeatedly until it decides it has enough evidence."""
    messages = [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": task},
    ]
    rounds = 0
    while rounds < MAX_ROUNDS:
        rounds += 1
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS
        )
        message = response.choices[0].message

        # No tool call means the agent has finished and written its findings
        if not message.tool_calls:
            return message.content

        messages.append(message)
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            log(f"Search: {args.get('query')}  (filter: {args.get('doc_type', 'none')})")
            try:
                result = search_documents(**args)
            except Exception as e:
                result = f"Search failed: {e}"
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    # Safety valve reached: ask for findings without further searching
    messages.append(
        {"role": "user", "content": "Stop searching now and give your findings."}
    )
    response = client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


def write_report(findings: dict) -> str:
    text = "\n\n".join(f"## {topic}\n{result}" for topic, result in findings.items())
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": REPORT_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


# ---------- Streamlit interface ----------
st.set_page_config(page_title="Canvassian Due Diligence", page_icon="⚖️")
st.title("Canvassian Due Diligence")
st.write(
    "An AI agent searches the evidence for each critical risk, "
    "then drafts a report for the Board."
)

if st.button("Run due diligence", type="primary"):
    findings = {}
    for topic, task in TOPICS.items():
        with st.status(f"Investigating: {topic}", expanded=False) as status:
            findings[topic] = investigate(task, log=st.write)
            status.update(label=f"Done: {topic}", state="complete")

    with st.spinner("Writing the Board report..."):
        report = write_report(findings)
    st.session_state["findings"] = findings
    st.session_state["report"] = report

if "report" in st.session_state:
    st.header("Board report")
    st.markdown(st.session_state["report"])
    st.download_button(
        "Download report (.md)",
        st.session_state["report"],
        file_name="canvassian_due_diligence_report.md",
    )
    with st.expander("Agent findings by topic"):
        for topic, result in st.session_state["findings"].items():
            st.subheader(topic)
            st.markdown(result)