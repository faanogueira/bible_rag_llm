import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# -----------------------------------------------------------------------------
# Caminhos
# -----------------------------------------------------------------------------

DIR_BASE      = Path(__file__).parent.resolve()
CAMINHO_PDF   = DIR_BASE / "data" / "biblia_arc.pdf"
PASTA_VETORES = DIR_BASE / "vetores_biblia"

load_dotenv(dotenv_path=DIR_BASE / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print(f"❌ GOOGLE_API_KEY não encontrada. Configure: {DIR_BASE / '.env'}")
    sys.exit(1)

MODELO_LLM        = "gemini-2.0-flash"
MODELO_EMBEDDINGS = "models/embedding-001"


# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------

TEMPLATE_PROMPT = """
Você é um assistente bíblico virtual de uma igreja evangélica, especializado
na Bíblia Sagrada versão Almeida Revista e Corrigida (ARC).

Responda com base EXCLUSIVAMENTE no contexto fornecido abaixo.
Seja claro, respeitoso e fiel ao texto bíblico.
Ao citar versículos, use o formato: Livro Capítulo:Versículo.
Se a resposta não estiver no contexto, informe educadamente.

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""

prompt = PromptTemplate(
    template=TEMPLATE_PROMPT,
    input_variables=["context", "question"]
)


# -----------------------------------------------------------------------------
# Pipeline RAG
# -----------------------------------------------------------------------------

def carregar_e_dividir_pdf():
    if not CAMINHO_PDF.exists():
        print(f"❌ PDF não encontrado em: {CAMINHO_PDF}")
        sys.exit(1)

    print("📖 Carregando PDF...")
    paginas = PyPDFLoader(str(CAMINHO_PDF)).load()
    print(f"   {len(paginas)} páginas carregadas.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    ).split_documents(paginas)

    print(f"   {len(chunks)} chunks gerados.")
    return chunks


def criar_banco_vetorial(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model=MODELO_EMBEDDINGS,
        google_api_key=GOOGLE_API_KEY
    )

    if PASTA_VETORES.exists() and any(PASTA_VETORES.iterdir()):
        print("⚡ Carregando banco vetorial do disco...")
        banco = Chroma(
            persist_directory=str(PASTA_VETORES),
            embedding_function=embeddings
        )
    else:
        print(f"🔄 Criando banco vetorial em: {PASTA_VETORES}")
        banco = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(PASTA_VETORES)
        )

    print(f"   {banco._collection.count()} vetores prontos.")
    return banco


def criar_chain(banco):
    llm = ChatGoogleGenerativeAI(
        model=MODELO_LLM,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True
    )

    retriever = banco.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )


# -----------------------------------------------------------------------------
# Chat
# -----------------------------------------------------------------------------

def iniciar_chat(chain):
    print("\n" + "=" * 65)
    print("  ✝️  ASSISTENTE BÍBLICO — Bíblia Sagrada ARC")
    print("  Developed by Fábio Nogueira")
    print("  Powered by Google Gemini + LangChain + ChromaDB")
    print("=" * 65)
    print("  Digite sua pergunta ou 'sair' para encerrar.")
    print("=" * 65 + "\n")

    while True:
        pergunta = input("🙏 Você: ").strip()

        if pergunta.lower() in ["sair", "exit", "quit", "q"]:
            print("\n✝️  Que Deus te abençoe! Até logo.\n")
            break

        if not pergunta:
            continue

        print("\n⏳ Buscando na Bíblia...\n")

        try:
            resultado   = chain.invoke({"query": pergunta})
            resposta    = resultado["result"]
            source_docs = resultado.get("source_documents", [])

            print(f"📖 Assistente:\n{resposta}")

            if source_docs:
                paginas = sorted(set(
                    doc.metadata.get("page", 0) + 1 for doc in source_docs
                ))
                print(f"\n📚 Fontes — páginas do PDF: {paginas}")

        except Exception as e:
            print(f"❌ Erro: {e}")

        print("\n" + "-" * 65 + "\n")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  Sistema RAG — Bíblia Sagrada")
    print(f"  Diretório: {DIR_BASE}")
    print("=" * 65)

    chunks = carregar_e_dividir_pdf()
    banco  = criar_banco_vetorial(chunks)

    print("\n🔗 Inicializando pipeline RAG...")
    chain = criar_chain(banco)
    print("   ✅ Pronto!\n")

    iniciar_chat(chain)
