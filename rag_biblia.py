import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.prompts import PromptTemplate


# -----------------------------------------------------------------------------
# Caminhos e configuração
# -----------------------------------------------------------------------------

DIR_BASE      = Path(__file__).parent.resolve()
DIR_DATA      = DIR_BASE / "data"
DIR_STORAGE   = DIR_BASE / "storage"

load_dotenv(dotenv_path=DIR_BASE / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print(f"❌ GOOGLE_API_KEY não encontrada. Configure: {DIR_BASE / '.env'}")
    sys.exit(1)

Settings.llm = GoogleGenAI(
    model="gemini-2.0-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001",
    api_key=GOOGLE_API_KEY,
)


# -----------------------------------------------------------------------------
# Prompt personalizado
# -----------------------------------------------------------------------------

TEMPLATE = (
    "Você é um assistente bíblico de uma igreja evangélica, especializado "
    "na Bíblia Sagrada versão Almeida Revista e Corrigida (ARC).\n\n"
    "Responda com base EXCLUSIVAMENTE no contexto abaixo. "
    "Seja claro, respeitoso e fiel ao texto bíblico. "
    "Ao citar versículos, use o formato Livro Capítulo:Versículo. "
    "Se a resposta não estiver no contexto, informe educadamente.\n\n"
    "Contexto:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Pergunta: {query_str}\n\n"
    "Resposta:"
)

prompt = PromptTemplate(TEMPLATE)


# -----------------------------------------------------------------------------
# Índice vetorial
# -----------------------------------------------------------------------------

def carregar_ou_criar_indice():
    if DIR_STORAGE.exists() and any(DIR_STORAGE.iterdir()):
        print("⚡ Carregando índice do disco...")
        storage_context = StorageContext.from_defaults(persist_dir=str(DIR_STORAGE))
        index = load_index_from_storage(storage_context)
        print("   Índice carregado.")
    else:
        if not DIR_DATA.exists() or not any(DIR_DATA.glob("*.pdf")):
            print(f"❌ Nenhum PDF encontrado em: {DIR_DATA}")
            sys.exit(1)

        print("📖 Carregando PDF e criando índice vetorial...")
        docs = SimpleDirectoryReader(str(DIR_DATA)).load_data()
        print(f"   {len(docs)} páginas carregadas.")

        index = VectorStoreIndex.from_documents(docs, show_progress=True)
        index.storage_context.persist(persist_dir=str(DIR_STORAGE))
        print(f"   ✅ Índice criado e salvo em: {DIR_STORAGE}")

    return index


# -----------------------------------------------------------------------------
# Chat
# -----------------------------------------------------------------------------

def iniciar_chat(query_engine):
    print("\n" + "=" * 65)
    print("  ✝️  ASSISTENTE BÍBLICO — Bíblia Sagrada ARC")
    print("  Powered by Google Gemini + LlamaIndex")
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
            resposta = query_engine.query(pergunta)

            print(f"📖 Assistente:\n{resposta}\n")

            fontes = getattr(resposta, "source_nodes", [])
            if fontes:
                paginas = sorted(set(
                    int(n.metadata.get("page_label", n.metadata.get("page", 0)))
                    for n in fontes
                    if n.metadata
                ))
                print(f"📚 Fontes — páginas do PDF: {paginas}")

        except Exception as e:
            print(f"❌ Erro: {e}")

        print("\n" + "-" * 65 + "\n")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  Sistema RAG — Bíblia Sagrada ARC")
    print("  LlamaIndex + Google Gemini")
    print(f"  Diretório: {DIR_BASE}")
    print("=" * 65 + "\n")

    index = carregar_ou_criar_indice()

    query_engine = index.as_query_engine(
        similarity_top_k=5,
        text_qa_template=prompt,
    )

    iniciar_chat(query_engine)
