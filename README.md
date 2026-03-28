# ✝️ Sistema RAG — Bíblia Sagrada ARC

> Chat inteligente com a Bíblia Sagrada (versão Almeida Revista e Corrigida)  
> utilizando **Recuperação Aumentada por Geração (RAG)** com **LangChain** e **Google Gemini**.

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura RAG](#arquitetura-rag)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Configuração da API Key](#configuração-da-api-key)
- [Execução](#execução)
- [Exemplos de Perguntas](#exemplos-de-perguntas)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Referências](#referências)

---

## Visão Geral

Este sistema permite que usuários façam perguntas em linguagem natural sobre a Bíblia Sagrada (ARC) e recebam respostas contextualizadas, fundamentadas exclusivamente no texto bíblico.

| Componente | Tecnologia | Função |
|---|---|---|
| **LLM** | Google Gemini 2.0 Flash | Geração das respostas |
| **Embeddings** | Google `embedding-001` | Vetorização dos trechos |
| **Banco vetorial** | ChromaDB (local) | Armazenamento e busca semântica |
| **Orquestração** | LangChain | Pipeline RAG completo |
| **Leitura do PDF** | PyPDF | Extração do texto da Bíblia |

---

## Arquitetura RAG

```
┌─────────────────────────────────────────────────────────────┐
│                    INDEXAÇÃO (1ª execução)                  │
│                                                             │
│  biblia_arc.pdf  →  Chunks (1000 chars)  →  Embeddings     │
│                                                  │          │
│                                             ChromaDB        │
│                                           (salvo em disco)  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  CONSULTA (toda execução)                   │
│                                                             │
│  Pergunta do usuário                                        │
│       │                                                     │
│       ▼                                                     │
│  Embedding da pergunta  →  Busca semântica no ChromaDB      │
│                                   │                         │
│                          Top 5 chunks relevantes            │
│                                   │                         │
│                          Prompt personalizado               │
│                          (contexto + pergunta)              │
│                                   │                         │
│                          Google Gemini (LLM)                │
│                                   │                         │
│                          Resposta contextualizada           │
└─────────────────────────────────────────────────────────────┘
```

> **Diferencial RAG:** o modelo responde apenas com base no texto da Bíblia ARC fornecida,  
> evitando respostas inventadas (alucinações) e garantindo fidelidade ao documento.

---

## Pré-requisitos

- Python **3.10+**
- PDF da Bíblia Sagrada versão ARC (arquivo: `biblia_arc.pdf`)
- Conta no [Google AI Studio](https://aistudio.google.com) (gratuita)

---

## Instalação

```bash
# 1. Clone ou baixe o projeto e entre na pasta
cd rag_biblia

# 2. Crie e ative o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instale as dependências
pip install langchain langchain-google-genai langchain-community \
            langchain-chroma pypdf python-dotenv chromadb
```

---

## Configuração da API Key

### 1. Obter a chave gratuitamente
Acesse [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey), faça login com sua conta Google e clique em **"Create API Key"**.

### 2. Configurar no projeto
Edite o arquivo `.env` na pasta do projeto:

```env
GOOGLE_API_KEY=sua_chave_aqui
```

> ⚠️ **Nunca compartilhe sua API Key** nem a inclua em repositórios públicos.  
> Adicione `.env` ao seu `.gitignore`.

---

## Execução

### 1. Coloque o PDF na pasta do projeto
```bash
# O arquivo deve se chamar exatamente:
biblia_arc.pdf
```

### 2. Execute o script
```bash
python rag_biblia.py
```

### O que acontece em cada execução:

| Situação | Comportamento |
|---|---|
| **1ª execução** | Processa o PDF, gera embeddings e salva o banco vetorial em `vetores_biblia/` |
| **Execuções seguintes** | Carrega o banco do disco — muito mais rápido |

---

## Exemplos de Perguntas

```
🙏 Você: Qual é o versículo mais famoso da Bíblia?
🙏 Você: Quais são as bem-aventuranças?
🙏 Você: O que Jesus disse sobre o amor ao próximo?
🙏 Você: Quais são os Dez Mandamentos?
🙏 Você: Fale sobre a criação do mundo no Gênesis
🙏 Você: O que Salomão diz sobre a sabedoria em Provérbios?
🙏 Você: Qual é o fruto do Espírito Santo em Gálatas?
```

Para encerrar o chat, digite: `sair`

---

## Estrutura do Projeto

```
rag_biblia/
├── rag_biblia.py          # Script principal
├── biblia_arc.pdf         # PDF da Bíblia ARC (adicionar manualmente)
├── .env                   # Chave de API (não versionar)
├── .gitignore             # Ignorar .env e vetores_biblia/
├── README.md              # Este arquivo
└── vetores_biblia/        # Banco vetorial ChromaDB (gerado na 1ª execução)
```

### `.gitignore` recomendado

```
.env
.venv/
vetores_biblia/
__pycache__/
*.pyc
```

---

## Referências

- [LangChain — Documentação oficial](https://docs.langchain.com)
- [Google AI Studio — API Key](https://aistudio.google.com/app/apikey)
- [langchain-google-genai — PyPI](https://pypi.org/project/langchain-google-genai)
- [ChromaDB — Banco vetorial](https://www.trychroma.com)
- [Bíblia Sagrada ARC — Versão online](https://www.bibliaonline.com.br/arc)

---

> **Atividade Processual 4** · Modelos Generativos · IPOG — Ciência de Dados
