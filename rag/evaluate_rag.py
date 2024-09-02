import json
from tqdm import tqdm
import os

# local package
# --------------------------------
from .src.normalize_text import normalize


# Run rag experiments
def run_rag_experiment(
    experiment_name="",
    query_data={},
    rag=None,
    verbose=False,
    results_dir="",
    is_ad_rag=False
) :
    """Run a rag experiment

    Args:
        experiment_name (str, optional): The experiment name of the rag pipeline. Defaults to "".
        query_data (dict): The multi hop rag query. Defaults to None.
        rag (_type_, optional): RAG pipeline. Defaults to None.
        verbose (bool, optional): Show the information of the results. Defaults to False.
        results_dir (str, optional): The result dir. Defaults to "".
        is_ad_rag (bool, optional): Whether to adrag the results. Defaults to False.
        
    #TODO: Tips 原数据有bug就是, 我在生成那里测试了但是不知道还会不会出问题
    for d in retrieval_save_list:
        if d['question_type'] == 'null_query':
            continue
        retrieved_lists.append([m['text'] for m in d['retrieval_list']])
        gold_lists.append([m['fact'] for m in d['gold_list']])
    
    """
    
    # Test retrieval quality
    retrieval_save_list = []
    for data in tqdm(query_data):
        
        query = data['query']
        
        if data['question_type'] == 'null_query':
            continue
        
        retrieval_list = []
        if is_ad_rag:
            nodes_score = rag.query(input=query)

            for ns in nodes_score:
                dic = {}
                dic['text'] = ns.get_content()
                dic['score'] = ns.get_score()
                retrieval_list.append(dic)
             
        else:
            nodes_score = rag.query(input=query)

            for ns in nodes_score:
                dic = {}
                dic['text'] = ns['text']
                dic['score'] = ns['score']
                retrieval_list.append(dic)
             
        save = {}
        save['query'] = data['query']   
        save['answer'] = data['answer']   
        save['question_type'] = data['question_type'] 
        save['retrieval_list'] = retrieval_list
        save['gold_list'] = data['evidence_list']   
        retrieval_save_list.append(save)
        
        print(save)
        import sys
        sys.exit()

    os.makedirs(results_dir, exist_ok=True)
    # Save the retrieval results
    save_file = os.path.join(results_dir,experiment_name)
    with open(f"{save_file}_retrieval.json", 'w') as json_file:
        json.dump(retrieval_save_list, json_file)
        
    # Evaluate the retrieval results
    
    retrieved_lists = []
    gold_lists  = []
    for d in retrieval_save_list:
        retrieved_lists.append([m['text'] for m in d['retrieval_list']])
        gold_lists.append([m['fact'] for m in d['gold_list']])
        
    metrics = calculate_metrics(retrieved_lists, gold_lists)
    
    if verbose:
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
        print('-'*20)
        
    with open(f"{save_file}_retrieval_metrics.json", 'w') as json_file:
        json.dump(metrics, json_file)
        
    return metrics
        
# Calculate the difference metric
def calculate_metrics(retrieved_lists, gold_lists):
    hits_at_10_count = 0
    map_at_10_list = []
    mrr_list = []

    for retrieved, gold in tqdm(zip(retrieved_lists, gold_lists), total=len(gold_lists)):
        hits_at_10_flag = False
        average_precision_sum = 0
        first_relevant_rank = None
        find_gold = []

        gold = [normalize(item) for item in gold]
        retrieved = [normalize(item) for item in retrieved]

        for rank, retrieved_item in enumerate(retrieved[:11], start=1):
            if any(gold_item in retrieved_item for gold_item in gold):
                if rank <= 10:
                    hits_at_10_flag = True
                    if first_relevant_rank is None:
                        first_relevant_rank = rank
                    # Compute precision at this rank for this query
                    count = 0
                    for gold_item in gold:
                        if gold_item in retrieved_item and not gold_item in find_gold:
                            count += 1
                            find_gold.append(gold_item)
                    precision_at_rank = count / rank
                    average_precision_sum += precision_at_rank

        # Calculate metrics for this query
        hits_at_10_count += int(hits_at_10_flag)
        map_at_10_list.append(average_precision_sum / min(len(gold), 10))
        mrr_list.append(1 / first_relevant_rank if first_relevant_rank else 0)

    # Calculate average metrics over all queries
    hits_at_10 = hits_at_10_count / len(gold_lists)
    map_at_10 = sum(map_at_10_list) / len(gold_lists)
    mrr_at_10 = sum(mrr_list) / len(gold_lists)

    return {
        'Hits@10': hits_at_10,
        'MAP@10': map_at_10,
        'MRR@10': mrr_at_10,
    }


        
        
        
      