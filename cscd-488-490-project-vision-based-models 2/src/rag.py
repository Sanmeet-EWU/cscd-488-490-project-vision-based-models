import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from src.keys import load_api_key

# THIS SCRIPT CREATES VECTOR STORAGE - IF .rag/vector_storage ALREADY EXISTS, NO NEED TO RUN THIS SCRIPT AGAIN
load_api_key("OPENAI_API_KEY")

PERSIST_DIR = "./rag/vector_storage"

def loadVectorStorage(path):
    dataset_file = "./rag/combined.txt" 
    
    if not os.path.exists(dataset_file):
        print(f"Dataset file '{dataset_file}' not found!")
        return
    
    index = load_and_index_dataset(dataset_file)


def load_and_index_dataset(file_path):   #creation or load of the vector index.
    try:
        if not os.path.exists(PERSIST_DIR):
            print("Loading documents and building the index...")

            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()   # read the dataset
            index = VectorStoreIndex.from_documents(documents)                      # convert the dataset into vector
            index.storage_context.persist(persist_dir=PERSIST_DIR)                  #Store ( check in same package)
            print("Index created and saved.")
        else:
            print("Loading existing index from storage...")
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)       #load the previosuly saved index
        return index
    except Exception as e:
        print(f"Error during indexing or loading: {e}")
        raise

loadVectorStorage(".rag/combined.txt")
