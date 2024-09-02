import argparse
import os
import json
import time
import logging
import re

from llm.toqd_with_IR import ToQDWithIR
from llm.evaluate_llm import run_toqd_with_IR_experiment
from llm.llm_api import LLMInterface
from rag.utils import load_config
from rag.rag import Retriever
from rag.src import slurm

openai_api_key = load_config()


def main(args):
    
    # Set up the basic LLM
    llm = LLMInterface(
        model_name=args.llm_name,
        api_key=openai_api_key,
        temperature=args.temperature,
        platform=args.platform,
        top_p=args.top_p
    )


    tmp = {
        "passages": args.passages,
        "passages_embeddings": args.passages_embeddings,
        "n_docs": args.n_docs,
        "validation_workers": args.validation_workers,
        "per_gpu_batch_size": args.per_gpu_batch_size,
        "save_or_load_index": args.save_or_load_index,
        "model_name_or_path": args.model_name_or_path,
        "cache_dir": args.cache_dir,
        "no_fp16": args.no_fp16,
        "question_maxlength": args.question_maxlength,
        "indexing_batch_size": args.indexing_batch_size,
        "projection_size": args.projection_size,
        "n_subquantizers": args.n_subquantizers,
        "n_bits": args.n_bits,
        "lang": args.lang,
        "dataset": args.dataset,
        "lowercase": args.lowercase,
        "normalize_text": args.normalize_text
    }

    
    
    
    retriever = Retriever(tmp)
    retriever.setup_retriever()
    

    toqd_with_ir = ToQDWithIR(
        llm=llm,
        retriever=retriever,
    )
    
    
    experiments = {
        'WikiMultiHopQA' : '2wiki_multi_hop_qa',  
    }
    
    with open(args.dataset_path,'r') as file:
        query_data = json.load(file)
    
    results_list = []
    for experiment_name,flag in experiments.items():
        save = {}
        total_accuracy, average_time, input_words_per_query, output_words_per_query, rounds_per_query = run_toqd_with_IR_experiment(
            experiment_name=experiment_name,
            query_data=query_data,
            verbose=args.verbose,
            toqd=toqd_with_ir,
            results_dir=args.results_dir,
            top_n=args.top_n,
        )
        
        save = {
            'experiment_name': experiment_name,
            'total_accuracy': total_accuracy,
            'average_time': average_time,
            'input_words_per_query': input_words_per_query,
            'output_words_per_query': output_words_per_query,
            'rounds_per_query': rounds_per_query,
        }
        
        results_list.append(save)
        
    all_results_path = os.path.join(args.results_dir,f"all_results.json")
    with open(all_results_path,'a') as json_file:
         json.dump(results_list, json_file)
    
    
if __name__ == '__main__':
    
    """
    llm = LLMInterface(
        model_name="gpt-3.5-turbo",
        api_key=openai_api_key,
        temperature=0,
        platform="openai",
        top_p=0.10
    )
    
    toqd_without_IR = ToQDWithoutIR(
        llm=llm,
    )
    
    print(toqd_without_IR.response("Which music group has more members, Bleeker or Bracket?",verbose=True))
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--llm_name",
        type=str,
        default="gpt-3.5-turbo",
        help="The large language model (LLM) to refine the query. Default the 'gpt-3.5-turbo'",
    )
    
    parser.add_argument(
        "--platform",
        type=str,
        default="openai",
        help="The platform for using the LLM. Default is 'openai'",
    )
    
    parser.add_argument(
        "--temperature",
        type=int,
        default=0,
        help="The temperature of the LLM to generate responses. Default is 0",
    )
    
    parser.add_argument(
        "--top_p",
        type=int,
        default=0.10,
        help="The top p of the LLM to generate responses. Default is 0.10",
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/wiki_multi_hop_qa/dev.json",
        help="The path to dataset directory",
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        #default="./results/wiki_multi_hop_qa/",
        default="./results/withIR/",
        help="The path to results",
    )
    
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Output the results",
    )
    
    parser.add_argument(
        "--passages", 
        type=str, 
        default="./db/wikipedia/psgs_w100.tsv", 
        help="Path to passages (.tsv file)"
    )
    
    parser.add_argument(
        "--passages_embeddings", 
        type=str, 
        default="./db/wikipedia/wikipedia_embeddings/*", 
        help="Glob path to encoded passages"
    )
    
    parser.add_argument(
        "--n_docs", 
        type=int, 
        default=100, 
        help="Number of documents to retrieve per questions"
    )
    
    parser.add_argument(
        "--top_n", 
        type=int, 
        default=1, 
        help="The retrieval documents"
    )
    
    
    parser.add_argument(
        "--validation_workers", 
        type=int, 
        default=32, 
        help="Number of parallel processes to validate results"
    )
    
    parser.add_argument(
        "--per_gpu_batch_size", 
        type=int, 
        default=64, 
        help="Batch size for question encoding"
    )
    
    parser.add_argument(
        "--save_or_load_index", 
        action="store_true", 
        default=True,
        help="If enabled, save index and load index if it exists"
    )
    
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="facebook/contriever-msmarco",
        help="path to directory containing model weights and config file"
    )
    
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default="./repo",
        help="path to directory containing model "
    )
    
    parser.add_argument(
        "--no_fp16", 
        action="store_true", 
        default=True,
        help="inference in fp32"
    )
    
    parser.add_argument(
        "--question_maxlength", 
        type=int, 
        default=512, 
        help="Maximum number of tokens in a question"
    )
    
    parser.add_argument(
        "--indexing_batch_size", 
        type=int, 
        default=1000000, 
        help="Batch size of the number of passages indexed"
    )
    
    parser.add_argument(
        "--projection_size", 
        type=int, 
        default=768
    )
    
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument("--lowercase", action="store_true", default=True,help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", default=True,help="normalize text")
    
    args = parser.parse_args()
    main(args)
    
    