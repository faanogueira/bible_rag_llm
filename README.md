# ✝️ Bible RAG — Chat com a Bíblia Sagrada ARC

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Core-7C3AED?style=flat)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.0%20Flash-4285F4?style=flat&logo=google&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-LlamaIndex%20%2B%20Gemini-8B5CF6?style=flat)

</div>

---

> *"A tua palavra é lâmpada para os meus pés e luz para o meu caminho."* — Salmos 119:105

O **Bible RAG** aplica a técnica de _Retrieval-Augmented Generation (RAG)_ sobre o maior manual de instruções para a vida humana já escrito — a **Bíblia Sagrada**. Com mais de **3.500 anos de história**, 66 livros, 1.189 capítulos e mais de 31.000 versículos, a Bíblia é o livro mais publicado, traduzido e lido de todos os tempos. Este projeto transforma esse vasto corpus em uma **base de conhecimento semântica**, permitindo que qualquer pessoa faça perguntas em linguagem natural e receba respostas precisas, contextualizadas e fiéis ao texto da versão **Almeida Revista e Corrigida (ARC)**.

Do ponto de vista técnico, o sistema implementa um pipeline RAG completo com **LlamaIndex**: o PDF da Bíblia é carregado, vetorizado com os embeddings do Google e indexado em um armazenamento local persistente. A cada consulta, os trechos mais semanticamente relevantes são recuperados e enviados ao **Google Gemini 2.0 Flash** como contexto, garantindo respostas fundamentadas exclusivamente no texto bíblico — sem alucinações, sem invenções.

**Stack utilizada:**

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.10+ |
| Orquestração RAG | LlamaIndex Core |
| Modelo de linguagem | Google Gemini 2.0 Flash |
| Embeddings | Google `embedding-001` |
| Índice vetorial | LlamaIndex VectorStoreIndex (local, persistente) |
| Variáveis de ambiente | python-dotenv |

---

## Visão Geral

O **Bible RAG** utiliza _Retrieval-Augmented Generation_ para garantir que todas as respostas sejam extraídas diretamente da Bíblia Sagrada ARC, eliminando alucinações e mantendo fidelidade ao texto sagrado.

**Destaques:**
- Busca semântica no texto completo da Bíblia ARC
- Citação de versículos no formato `Livro Capítulo:Versículo`
- Índice vetorial persistente — o PDF é processado apenas na primeira execução
- Interface de chat interativa no terminal

---

## Arquitetura

```
bible_rag_llm/
├── rag_biblia.py          # Pipeline RAG principal
├── requirements.txt       # Dependências do projeto
├── .env                   # Chave de API (não versionada)
├── .gitignore
├── README.md
├── data/
│   └── biblia_arc.pdf     # Base de conhecimento
└── storage/               # Índice vetorial (gerado na 1ª execução)
```

```
[1ª EXECUÇÃO — Indexação]

biblia_arc.pdf
     │
     ▼
SimpleDirectoryReader → páginas carregadas
     │
     ▼
GoogleGenAIEmbedding (embedding-001)
     │
     ▼
VectorStoreIndex → storage/


[CONSULTA — A cada pergunta]

Pergunta → Embedding → Busca semântica (top 5 nós)
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

pip install -r requirements.txt
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
python3 rag_biblia.py
```

| Execução | Comportamento | Tempo estimado |
|---|---|---|
| **1ª vez** | Indexa o PDF e salva o índice em `storage/` | 5–15 min |
| **Demais** | Carrega o índice do disco e inicia o chat | < 30 seg |

**Exemplos de perguntas:**

```
🙏 Você: Quais são as bem-aventuranças?
🙏 Você: O que Jesus disse sobre o amor ao próximo?
🙏 Você: Qual é o fruto do Espírito Santo em Gálatas 5?
🙏 Você: O que diz o Salmo 23?
🙏 Você: Fale sobre a criação do mundo no Gênesis
```

Para encerrar, digite `sair`.

---

## Tecnologias

| Tecnologia | Papel |
|---|---|
| [LlamaIndex Core](https://docs.llamaindex.ai) | Orquestração do pipeline RAG |
| [Google Gemini 2.0 Flash](https://ai.google.dev) | Geração de respostas |
| [Google embedding-001](https://ai.google.dev) | Vetorização dos textos |
| [python-dotenv](https://pypi.org/project/python-dotenv) | Gerenciamento de variáveis de ambiente |

---

## Referências

- [LlamaIndex Docs](https://docs.llamaindex.ai)
- [Google AI Studio](https://aistudio.google.com)
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
