import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from keys import load_api_key

# THIS SCRIPT CREATES VECTOR STORAGE - IF .rag/vector_storage ALREADY EXISTS, NO NEED TO RUN THIS SCRIPT AGAIN
load_api_key("OPENAI_API_KEY")

PERSIST_DIR = "./rag/vector_storage"

def loadVectorStorage(path):
    dataset_file = "./rag/combined.txt" 
    
    if not os.path.exists(dataset_file):
        print(f"Dataset file '{dataset_file}' not found!")
        return
    
    index = load_and_index_dataset(dataset_file)

def load_and_index_dataset(file_path):
    try:
        embed_model = OpenAIEmbedding(model="text-embedding-ada-002")  # ✅ Define embedding model here

        if os.path.exists(PERSIST_DIR):
            print("🔄 Loading existing index from storage...")
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)  # ✅ Load index normally
            print("✅ Existing index loaded successfully!")

        else:
            print("🚀 Creating new index...")

            # ✅ Load & Chunk Documents
            node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=100)
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            nodes = node_parser.get_nodes_from_documents(documents)

            # ✅ Embed Documents & Create Index
            index = VectorStoreIndex(nodes, embed_model=embed_model)
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            print("✅ Index created and saved.")

        print(f"🔍 Total indexed documents: {len(index.docstore.docs)}")
        return index

    except Exception as e:
        print(f"❌ Error during indexing: {e}")
        raise
loadVectorStorage(".rag/combined.txt")
