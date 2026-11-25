# MaintiDoc · Partie I

Ce prototype interroge un PDF contenant du texte, des figures et des tableaux. Il fournit une réponse avec des références et permet d’ouvrir les éléments utilisés. Le document de démonstration est *Attention Is All You Need*.

| Tâche du tableau | Réalisation |
| --- | --- |
| 1. Choix du PDF | Article de 11 pages avec deux figures et trois tableaux numérotés. |
| 2. Extraction et indexation | PyMuPDF lit le document. LlamaIndex découpe le texte. CLIP associe les figures à des étiquettes visuelles et produit leurs vecteurs. MiniLM multilingue vectorise le texte, les légendes enrichies et les tableaux. ChromaDB conserve les deux index. |
| 3. Recherche et génération | Recherche par similarité dans ChromaDB et réponse en français avec Qwen 2.5 1.5B via Ollama. Chaque citation renvoie à un élément réellement retrouvé. |
| 4. Interface Streamlit | Formulaire, mode de recherche, réponse et volet de sources affichant le texte, les images ou les tableaux. |
| 5. Agent LangChain | Qwen choisit une modalité ou demande une clarification. LangGraph exécute l’outil LangChain `rechercher_pdf` et le résultat affiche ce chemin. |

## Exécuter sous Windows

Dans PowerShell, depuis le dossier `maintidoc`, avec Python 3.12, `uv` et Ollama installés, et le service Ollama démarré :

```powershell
uv venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
ollama pull qwen2.5:1.5b

.\.venv\Scripts\python.exe ingest.py "$env:USERPROFILE/Downloads/attention is all you need.pdf"
.\.venv\Scripts\python.exe rag.py index
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Ouvrir `http://127.0.0.1:8502`. Les poids MiniLM multilingue et CLIP quantifiés sont téléchargés au premier indexage et exécutés localement sur CPU. Le manifeste `artifacts/elements.jsonl` contient les références. Les images extraites sont dans `artifacts/images/` et ChromaDB dans `artifacts/chroma/`. Après avoir changé de document ou de modèle d’embeddings, reconstruire l’index et relancer Streamlit.

Pour utiliser un autre modèle Ollama installé, définir `MAINTIDOC_CHAT_MODEL` avant de lancer l’application. Pour changer de modèle d’embeddings, définir `MAINTIDOC_EMBED_MODEL` et reconstruire l’index.

## Démontrer les trois modalités

| Question | Élément à retrouver |
| --- | --- |
| Quelle tâche expérimentale les auteurs présentent-ils dans le résumé ? | Le résumé de la page 1 mentionne la traduction automatique. |
| Que montre la figure 2 ? | Page 4. L’attention par produit scalaire mis à l’échelle et l’attention multi-tête. |
| Quel score BLEU le Transformer (big) obtient-il en anglais-allemand d’après le tableau 2 ? | Page 8. La colonne `BLEU EN-DE` contient **28,4**. |
| Quel prix ce PDF donne-t-il pour un billet de train entre Rabat et Fès ? | Le document ne fournit pas cette information. |

Le mode **RAG direct** teste la recherche et la génération. Le mode **Agent LangChain** affiche le choix de l’outil. La même vérification peut être exécutée en ligne de commande.

```powershell
.\.venv\Scripts\python.exe rag.py search 'Que montre la figure 2 ?' --modality figure
.\.venv\Scripts\python.exe rag.py ask 'Quel score BLEU le Transformer (big) obtient-il en anglais-allemand d’après le tableau 2 ?' --agent
```

## Limites du prototype

- Les trois tableaux de l’article utilisent des repères de mise en page vérifiés, sélectionnés d’après l’empreinte du PDF. Sur d’autres PDF textuels, le lecteur utilise la détection automatique des tableaux et les images intégrées. Les PDF scannés et les figures entièrement vectorielles demandent un traitement supplémentaire.
- CLIP sélectionne une étiquette parmi des descriptions candidates et indexe chaque figure. La légende originale reste la source pour les détails. Les captures servent à contrôler les formules, indices et cellules fusionnées.
- Les références sont validées contre les résultats récupérés. Pour vérifier une affirmation, ouvrir l’extrait cité dans l’interface. Sur CPU, Ollama peut prendre plus d’une minute sur une question à plusieurs sources.
