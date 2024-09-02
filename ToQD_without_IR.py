import argparse
import os
import json
import time
import logging
import re

from llm.toqd import ToQDWithoutIR
from llm.evaluate_llm import run_toqd_without_IR_experiment
from llm.llm_api import LLMInterface
from rag.utils import load_config

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
    
    # Set up ToQD without IR
    toqd_without_ir = ToQDWithoutIR(
        llm=llm,
    )

    experiments = {
        'WikiMultiHopQA' : '2wiki_multi_hop_qa',    
    }
    
    with open(args.dataset_path,'r') as file:
        query_data = json.load(file)
    
    results_list = []
    for experiment_name,flag in experiments.items():
        save = {}
        total_accuracy, average_time = run_toqd_without_IR_experiment(
            experiment_name=experiment_name,
            query_data=query_data,
            verbose=args.verbose,
            toqd=toqd_without_ir,
            results_dir=args.results_dir,
        )
        
        save = {
            'experiment_name': experiment_name,
            'total_accuracy': total_accuracy,
            'average_time': average_time,
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
        default="./results/withoutIR/",
        help="The path to results",
    )
    
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Output the results",
    )
    
    args = parser.parse_args()
    main(args)