# ✝️ Bible RAG — Chat com a Bíblia Sagrada ARC

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.0%20Flash-4285F4?style=flat&logo=google&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Local-FF6B35?style=flat)

**Sistema RAG para consulta inteligente à Bíblia Sagrada versão ARC.**  
Perguntas em linguagem natural com respostas fundamentadas exclusivamente no texto bíblico.

</div>

---

## Visão Geral

O **Bible RAG** utiliza a técnica de _Retrieval-Augmented Generation_ para garantir que todas as respostas sejam extraídas diretamente da Bíblia Sagrada ARC, eliminando alucinações e mantendo fidelidade ao texto sagrado.

**Destaques:**
- Busca semântica no texto completo da Bíblia ARC
- Citação de versículos no formato `Livro Capítulo:Versículo`
- Banco vetorial persistente — o PDF é processado apenas na primeira execução
- Interface de chat interativa no terminal

---

## Arquitetura

```
bible_rag_llm/
├── rag_biblia.py          # Pipeline RAG principal
├── .env                   # Chave de API (não versionada)
├── .gitignore
├── README.md
├── data/
│   └── biblia_arc.pdf     # Base de conhecimento
└── vetores_biblia/        # Banco ChromaDB (gerado na 1ª execução)
```

```
[1ª EXECUÇÃO — Indexação]

biblia_arc.pdf
     │
     ▼
PyPDFLoader → chunks (1000 chars, overlap 200)
     │
     ▼
GoogleGenerativeAIEmbeddings (embedding-001)
     │
     ▼
ChromaDB → vetores_biblia/


[CONSULTA — A cada pergunta]

Pergunta → Embedding → Busca semântica (top 5 chunks)
     │
     ▼
Prompt contextualizado → Gemini 2.0 Flash → Resposta + fontes
```

---

## Instalação

**Pré-requisitos:** Python 3.10+, conta [Google AI Studio](https://aistudio.google.com/app/apikey) (gratuita)

```bash
git clone https://github.com/faanogueira/bible-rag-langchain-gemini.git
cd bible-rag-langchain-gemini

python3 -m venv .venv
source .venv/bin/activate

pip install langchain langchain-google-genai langchain-community \
            langchain-chroma pypdf python-dotenv chromadb
```

---

## Configuração

**1. API Key do Google Gemini**

Acesse [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey), gere uma chave gratuita e configure o `.env`:

```env
GOOGLE_API_KEY=sua_chave_aqui
```

**2. PDF da Bíblia**

Coloque o arquivo na pasta `data/`:

```
bible_rag_llm/
└── data/
    └── biblia_arc.pdf
```

---

## Execução

```bash
python rag_biblia.py
```

| Execução | Comportamento | Tempo estimado |
|---|---|---|
| **1ª vez** | Indexa o PDF e salva o banco vetorial | 5–15 min |
| **Demais** | Carrega o banco do disco e inicia o chat | < 30 seg |

**Exemplos de perguntas:**

```
🙏 Você: Quais são as bem-aventuranças?
🙏 Você: O que Jesus disse sobre o amor ao próximo?
🙏 Você: Qual é o fruto do Espírito Santo em Gálatas 5?
🙏 Você: O que diz o Salmo 23?
🙏 Você: Fale sobre a criação do mundo no Gênesis
```

---

## Tecnologias

| Tecnologia | Papel |
|---|---|
| [LangChain](https://docs.langchain.com) | Orquestração do pipeline RAG |
| [Google Gemini 2.0 Flash](https://ai.google.dev) | Geração de respostas |
| [Google embedding-001](https://ai.google.dev) | Vetorização dos textos |
| [ChromaDB](https://www.trychroma.com) | Banco de vetores local |
| [PyPDF](https://pypdf.readthedocs.io) | Extração do PDF |

---

## Referências

- [LangChain Docs](https://docs.langchain.com)
- [Google AI Studio](https://aistudio.google.com)
- [ChromaDB Docs](https://docs.trychroma.com)
- [Bíblia ARC Online](https://www.bibliaonline.com.br/arc)

---

## 👤 Autor

<!-- Início da seção "Contato" -->
<div>
  <p>Developed by <b>Fábio Nogueira</b></p>
</div>
<p>
<a href="https://www.linkedin.com/in/faanogueira/" target="_blank"><img style="padding-right: 10px;" src="https://img.icons8.com/?size=100&id=13930&format=png&color=000000" target="_blank" width="80"></a>
<a href="https://github.com/faanogueira" target="_blank"><img style="padding-right: 10px;" src="https://img.icons8.com/?size=100&id=AZOZNnY73haj&format=png&color=000000" target="_blank" width="80"></a>
<a href="https://api.whatsapp.com/send?phone=5571983937557" target="_blank"><img style="padding-right: 10px;" src="https://img.icons8.com/?size=100&id=16713&format=png&color=000000" target="_blank" width="80"></a>
<a href="mailto:faanogueira@gmail.com"><img style="padding-right: 10px;" src="https://img.icons8.com/?size=100&id=P7UIlhbpWzZm&format=png&color=000000" target="_blank" width="80"></a> 
</p>
<!-- Fim da seção "Contato" -->

---

<div align="center">

*"A tua palavra é lâmpada para os meus pés e luz para o meu caminho." — Salmos 119:105*

</div>
