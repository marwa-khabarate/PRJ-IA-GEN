"""Interface Streamlit de la Partie I."""

import pandas as pd
import streamlit as st

from rag import ask
from retrieval import CorpusIndex, source_title


st.set_page_config(page_title="MaintiDoc · RAG multimodal", page_icon="📄", layout="centered")


@st.cache_resource(show_spinner=False)
def load_index():
    return CorpusIndex().open()


def show_source(index, record):
    heading = source_title(record)
    if "id" in record:
        heading = f"[{record['id']}] {heading}"
    with st.expander(heading, expanded=record["type"] != "text"):
        st.caption(record["source"])
        if record.get("caption"):
            st.write(record["caption"])
        if "asset" in record:
            st.image(str(index.asset(record)), width="stretch")
        if record["type"] == "text":
            st.text(record["text"])
        elif record["type"] == "figure" and record.get("clip_label"):
            st.caption(f"Étiquette CLIP · {record['clip_label']}")
        elif record["type"] == "table":
            header, *rows = record["rows"]
            columns = [f"{name or 'Colonne'} ({i + 1})" for i, name in enumerate(header)]
            st.dataframe(pd.DataFrame(rows, columns=columns), hide_index=True, width="stretch")


st.title("MaintiDoc")
st.caption("Recherche documentaire · Partie I")

try:
    corpus = load_index()
except (FileNotFoundError, ValueError) as error:
    st.error(f"Le corpus local est indisponible. {error}")
    st.code("python rag.py index", language="powershell")
    st.stop()

st.markdown(f"**Document** · {corpus.document_name}")
with st.form("question", border=True):
    question = st.text_area("Votre question", max_chars=2000,
                            placeholder="Quel score BLEU le Transformer (big) obtient-il en anglais-allemand, d’après le tableau 2 ?")
    mode = st.radio("Mode de recherche", ("Agent LangChain", "RAG direct"), horizontal=True)
    submitted = st.form_submit_button("Interroger le PDF", type="primary")

if submitted:
    st.session_state.pop("response", None)
    if not question.strip():
        st.warning("Saisissez une question sur le document.")
    else:
        try:
            with st.spinner("Recherche des sources et rédaction de la réponse…"):
                st.session_state.response = ask(corpus, question, agentic=mode == "Agent LangChain")
        except Exception as error:
            st.error(f"La requête a échoué. Vérifiez le service Ollama et les modèles installés. {error}")

if response := st.session_state.get("response"):
    st.subheader("Réponse")
    st.write(response["answer"])
    st.caption(f"{response['route']} · {response['model']}")
    sources = response["sources"] or response["retrieved"]
    if sources:
        st.subheader("Sources citées" if response["sources"] else "Éléments retrouvés")
        for source in sources:
            show_source(corpus, source)
