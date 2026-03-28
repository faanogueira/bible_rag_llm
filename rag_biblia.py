# =============================================================================
# Atividade Processual 4 — Sistema RAG para Chat com a Bíblia Sagrada (ARC)
#
# Descrição:
#   Sistema de Recuperação Aumentada por Geração (RAG) que permite ao usuário
#   fazer perguntas em linguagem natural sobre a Bíblia Sagrada (versão ARC)
#   e receber respostas contextualizadas com base no documento PDF.
#
# Tecnologias:
#   - LangChain    : orquestração do pipeline RAG
#   - Google Gemini: modelo de linguagem (LLM) e embeddings
#   - ChromaDB     : banco de vetores local (persistente)
#   - PyPDF        : leitura do PDF da Bíblia
#
# Pré-requisitos:
#   pip install langchain langchain-google-genai langchain-community
#               langchain-chroma pypdf python-dotenv chromadb
#
# Uso:
#   1. Coloque o PDF da Bíblia ARC na mesma pasta com o nome: biblia_arc.pdf
#   2. Configure sua GOOGLE_API_KEY no arquivo .env
#   3. Execute: python rag_biblia.py
# =============================================================================

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


# =============================================================================
# CONFIGURAÇÃO INICIAL
# =============================================================================

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ GOOGLE_API_KEY não encontrada. Configure o arquivo .env")
    sys.exit(1)

# Caminho do PDF da Bíblia (deve estar na mesma pasta do script)
CAMINHO_PDF    = Path(__file__).parent / "biblia_arc.pdf"

# Pasta onde o banco vetorial será salvo (persistente entre execuções)
PASTA_VETORES  = Path(__file__).parent / "vetores_biblia"

# Modelos do Google Gemini
MODELO_LLM        = "gemini-2.0-flash"          # Modelo de geração de texto
MODELO_EMBEDDINGS = "models/embedding-001"       # Modelo de embeddings


# =============================================================================
# PROMPT PERSONALIZADO PARA CONTEXTO EVANGÉLICO
# =============================================================================

TEMPLATE_PROMPT = """
Você é um assistente bíblico virtual de uma igreja evangélica, especializado
na Bíblia Sagrada versão Almeida Revista e Corrigida (ARC).

Responda à pergunta do usuário com base EXCLUSIVAMENTE no conteúdo da Bíblia
fornecido abaixo. Seja claro, respeitoso e fiel ao texto bíblico.

Ao citar versículos, use o formato: Livro Capítulo:Versículo
(exemplo: João 3:16 — "Porque Deus amou o mundo...")

Se a resposta não estiver no contexto fornecido, diga educadamente que não
encontrou essa informação na base de dados atual.

Contexto da Bíblia:
{context}

Pergunta do usuário:
{question}

Resposta:
"""

prompt_personalizado = PromptTemplate(
    template=TEMPLATE_PROMPT,
    input_variables=["context", "question"]
)


# =============================================================================
# FUNÇÕES DO PIPELINE RAG
# =============================================================================

def verificar_pdf():
    """Verifica se o PDF da Bíblia existe no diretório esperado."""
    if not CAMINHO_PDF.exists():
        print(f"\n❌ PDF não encontrado em: {CAMINHO_PDF}")
        print("   Coloque o arquivo 'biblia_arc.pdf' na mesma pasta do script.")
        sys.exit(1)
    print(f"✅ PDF encontrado: {CAMINHO_PDF.name}")


def carregar_e_dividir_pdf():
    """
    Carrega o PDF e divide o texto em chunks para indexação.

    O RecursiveCharacterTextSplitter divide o texto preservando parágrafos
    e versículos, com sobreposição para não perder contexto entre chunks.
    """
    print("\n📖 Carregando e processando o PDF da Bíblia...")

    # Carrega todas as páginas do PDF
    loader = PyPDFLoader(str(CAMINHO_PDF))
    paginas = loader.load()
    print(f"   {len(paginas)} páginas carregadas.")

    # Divide em chunks menores para indexação vetorial
    # chunk_size: tamanho de cada pedaço em caracteres
    # chunk_overlap: sobreposição entre chunks (preserva contexto entre versículos)
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = divisor.split_documents(paginas)
    print(f"   {len(chunks)} chunks gerados para indexação.")
    return chunks


def criar_banco_vetorial(chunks):
    """
    Cria ou carrega o banco de vetores ChromaDB com os embeddings do documento.

    Na primeira execução: processa todos os chunks e salva em disco.
    Nas execuções seguintes: carrega o banco já existente (muito mais rápido).
    """
    # Inicializa o modelo de embeddings do Google
    embeddings = GoogleGenerativeAIEmbeddings(
        model=MODELO_EMBEDDINGS,
        google_api_key=GOOGLE_API_KEY
    )

    # Verifica se já existe um banco vetorial salvo em disco
    if PASTA_VETORES.exists() and any(PASTA_VETORES.iterdir()):
        print("\n⚡ Banco vetorial já existe — carregando do disco (modo rápido)...")
        banco_vetorial = Chroma(
            persist_directory=str(PASTA_VETORES),
            embedding_function=embeddings
        )
        print(f"   Banco carregado com {banco_vetorial._collection.count()} vetores.")
    else:
        print("\n🔄 Criando banco vetorial pela primeira vez (pode demorar alguns minutos)...")
        banco_vetorial = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(PASTA_VETORES)
        )
        print(f"   ✅ {banco_vetorial._collection.count()} vetores criados e salvos em disco.")

    return banco_vetorial


def criar_chain_rag(banco_vetorial):
    """
    Monta o pipeline RAG completo:
    Pergunta → Recuperação de contexto → LLM → Resposta
    """
    # Inicializa o modelo de geração de texto (LLM)
    llm = ChatGoogleGenerativeAI(
        model=MODELO_LLM,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,       # Baixo para respostas mais fiéis ao texto bíblico
        convert_system_message_to_human=True
    )

    # Configura o retriever — busca os 5 chunks mais relevantes por pergunta
    retriever = banco_vetorial.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Monta a chain RAG com prompt personalizado
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",            # Insere todos os chunks no contexto do LLM
        retriever=retriever,
        return_source_documents=True,  # Retorna os trechos usados como fonte
        chain_type_kwargs={"prompt": prompt_personalizado}
    )

    return chain


def exibir_fontes(source_docs):
    """Exibe as páginas do PDF usadas como fonte para a resposta."""
    paginas_usadas = sorted(set(
        doc.metadata.get("page", "?") + 1
        for doc in source_docs
    ))
    print(f"\n📚 Fontes consultadas — páginas do PDF: {paginas_usadas}")


# =============================================================================
# INTERFACE DE CHAT NO TERMINAL
# =============================================================================

def iniciar_chat(chain):
    """
    Loop principal do chat — mantém a conversa até o usuário digitar 'sair'.
    """
    print("\n" + "=" * 65)
    print("  ✝️  ASSISTENTE BÍBLICO — Bíblia Sagrada ARC")
    print("  Powered by Google Gemini + LangChain")
    print("=" * 65)
    print("  Digite sua pergunta sobre a Bíblia e pressione Enter.")
    print("  Para encerrar, digite: sair")
    print("=" * 65 + "\n")

    while True:
        # Lê a pergunta do usuário
        pergunta = input("🙏 Você: ").strip()

        # Verifica comandos de saída
        if pergunta.lower() in ["sair", "exit", "quit", "q"]:
            print("\n✝️  Que Deus te abençoe! Até logo.\n")
            break

        # Ignora entradas vazias
        if not pergunta:
            continue

        print("\n⏳ Buscando na Bíblia...\n")

        try:
            # Executa o pipeline RAG
            resultado = chain.invoke({"query": pergunta})

            resposta      = resultado["result"]
            source_docs   = resultado.get("source_documents", [])

            # Exibe a resposta
            print(f"📖 Assistente:\n{resposta}")

            # Exibe as fontes utilizadas
            if source_docs:
                exibir_fontes(source_docs)

        except Exception as e:
            print(f"❌ Erro ao processar a pergunta: {e}")

        print("\n" + "-" * 65 + "\n")


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("  Sistema RAG — Bíblia Sagrada ARC")
    print("  LangChain + Google Gemini + ChromaDB")
    print("=" * 65)

    # Etapa 1: Verificar se o PDF existe
    verificar_pdf()

    # Etapa 2: Carregar e dividir o PDF em chunks
    chunks = carregar_e_dividir_pdf()

    # Etapa 3: Criar ou carregar o banco vetorial
    banco_vetorial = criar_banco_vetorial(chunks)

    # Etapa 4: Montar o pipeline RAG
    print("\n🔗 Inicializando pipeline RAG...")
    chain = criar_chain_rag(banco_vetorial)
    print("   ✅ Pipeline pronto!")

    # Etapa 5: Iniciar o chat interativo
    iniciar_chat(chain)
