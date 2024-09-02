# Transformers
import pandas as pd
from tqdm import tqdm
import os
    
from dotenv import load_dotenv

# langchain
from llama_index.core import Document

from rag.src.normalize_text import normalize
from typing import Any, Generator, List, Optional

# Load the huggingface model, openai environment
# ------------------------------------------------------------
def load_hf_model(
    model_name="",
    repo="",
):
    """
    Load the model/pipeline from the huggingface community repository.
    If is_embedding is True the return embedding model,else return the pipeline of summarization 

    Args:
        model_name (str, optional): model_name. Defaults to "".
        repo (str, optional): the repo of model. Defaults to "".
        is_embedding (bool, optional): . Defaults to False.

    Raises:
        ValueError: _description_
    """
    
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    model = HuggingFaceEmbedding(
        model_name=model_name,
        cache_folder=repo,
        device='cuda:0',
        normalize=False
    )
    
    return model
            
    


def load_config():
    
    
    # Load environment variables from a .env file
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')

    return api_key
# loader functions
class Loader():
    """
    Load function for the wiki/multi_hop functions
    """
    
    def __init__(
        self,
        is_wiki: Optional[bool] = False,
    ) -> None:
        """Initialize with arguments."""
        super().__init__()
        self.is_wiki = is_wiki
        
        
    def load_data(self, input_file: str) -> List[Document]:
        """Load data from the input file."""
        
        if self.is_wiki:
            return self.__load_wiki_corpus(input_file=input_file)
        else:
            return self.__load_multi_hop_rag_corpus(input_file=input_file)
            
    def __load_multi_hop_rag_corpus(
        self,
        input_file: str
    ) -> List[Document]:

        df = pd.read_json(input_file)
        documents = [Document(text=normalize(content)) for content in df['body']]
            
        return documents
    
    def __load_wiki_corpus(
        self,
        input_file: str
    ) -> List[Document]:
        
        df = pd.read_csv(input_file, sep="\t")
        
        current_title = None
        current_texts = []
        documents = []
        
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
            title = row['title']
            text = row['text']
        
            text = normalize(text)
            
            # Check if we've moved to a new title or the DataFrame begins
            if title != current_title and current_title is not None:
                # Process the accumulated texts for the previous title
                documents.append(
                    Document(
                        text=' '.join(current_texts)
                    )
                )
                current_texts = []
                
            # Update the current title and accumulate text
            current_title = title
            current_texts.append(text)
        
        # Handle any remaining texts for the last title
        if current_texts:
            documents.append(
                Document(
                    text=' '.join(current_texts)
                )
            )
            
        return documents