import regex
import string
from tqdm import tqdm
import json
import os
import re
import time

# Over extract matched
import re
import string
import spacy

nlp = spacy.load('en_core_web_lg')  # Load the SpaCy model for entity extraction

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', '', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_entities(text, entity_types={'PERSON', 'ORG', 'GPE', 'LOC', 'DATE'}):
    """Extracts specified entity types from the text using SpaCy."""
    doc = nlp(text)
    return {ent.text for ent in doc.ents if ent.label_ in entity_types}

def match_or_not(prediction, ground_truth):
    norm_predict = normalize_answer(prediction)
    norm_answer = normalize_answer(ground_truth)
    
    # Split the normalized texts into sets of words
    set_predict = set(norm_predict.split())
    set_answer = set(norm_answer.split())
    
    if set_answer <= set_predict:
        # Check if all elements of the ground truth set are in the prediction set
        return set_answer <= set_predict
    
    entities_predict = extract_entities(norm_predict)
    entities_truth = extract_entities(norm_answer)
    numbers_predict = set(re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', norm_predict))
    numbers_truth = set(re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', norm_answer))

    # Initialize match variables based on presence of entities and numbers
    entity_match = entities_truth <= entities_predict if entities_truth and entities_predict else False
    number_match = numbers_truth == numbers_predict if numbers_truth and numbers_predict else False
    
    # Enhanced matching logic
    if entities_truth or numbers_truth:
        return (entity_match and number_match) if (entities_truth and numbers_truth) else entity_match or number_match
    return False

# Run self-verify experiment
def run_self_verify_experiment(
    experiment_name="",
    query_data={},
    retriever=None,
    llm=None,
    prompt_template=None,
    verbose=False,
    results_dir="",
    top_n=2,
    is_rag=False,
    is_cot=False
):
    """Run a self verification experiment 

    Args:
        experiment_name (str, optional): The experiment name of the rag pipeline. Defaults to "".
        query_data (dict): The multi hop rag query. Defaults to None.
        rag (_type_, optional): RAG pipeline. Defaults to None.
        verbose (bool, optional): Show the information of the results. Defaults to False.
        results_dir (str, optional): The result dir. Defaults to "".
        is_ad_rag (bool, optional): Whether to adrag the results. Defaults to False.
        llm (str, optional): The large language model (LLM) to answer the query.
        prompt_template (_type_, optional): The prompt template. Defaults to None.
        is_cot (bool, optional): Whether to use COT. Defaults to False.
    """
    
    results_list = []
    # Test the self verification validation
    
    count = 0
    total_count = 0
    no_count = 0
    yes_count = 0
    total_correct = 0
    no_correct = 0
    yes_correct = 0
    
    for data in tqdm(query_data):

        
        question = data['question']
        
        retrieval_list = []
        
        try:
            if is_rag:
            
                retrieval_results = retriever.search_document(question, top_n=top_n,verbose=False)
            
                context = '\n\n'.join(entry['text'] for entry in retrieval_results)
            
                ans = llm.response(
                        input= prompt_template.load(
                            node_type="general",
                            template_type="rag",
                            question=question,
                            context=context,
                        )
                    )
            
                ver = llm.response(
                        input= prompt_template.load(
                            node_type="self-verify",
                            template_type="rag",
                            question=question,
                            context=context,
                        )
                    )
            
                retrieval_list.append(retrieval_results)
            
            
        
            else:
            #is_rag = re.match(r'^NO!', ver) is not None
            
                if is_cot:
                    ans = llm.response(
                        input= prompt_template.load(
                            node_type="general",
                            template_type="CoT",
                            question=question,
                        )
                    )
                
                    ver = llm.response(
                        input=prompt_template.load(
                            node_type="self-verify",
                            template_type="CoT",
                            question=question,
                        )
                    )
                else:
                    ans = llm.response(
                        input= prompt_template.load(
                            node_type="general",
                            template_type="original",
                            question=question,
                        )
                    )
                
                    ver = llm.response(
                        input=prompt_template.load(
                            node_type="self-verify",
                            template_type="original",
                            question=question,
                        )
                    )
                    
        except Exception as e:
            import time
            time.sleep(120)
            
        is_no = re.match(r'^NO!', ver) is not None
        save = {
            'count' : count,
            'question' : data['question'],
            'response' : ans,
            'ver' : ver,
            'flag' : is_no,
            'gold_answer' : data['answer'],
            'retrieval_list' : retrieval_list
        }
            
        match_result = match_or_not(ans, data['answer'])
        results_list.append(save)
            
        total_count += 1
        total_correct += match_result

        if is_no:
            no_count += 1
            no_correct += match_result
        else:
            yes_count += 1
            yes_correct += match_result
        
        
            
    os.makedirs(results_dir, exist_ok=True)
    # Save the retrieval results
    save_file = os.path.join(results_dir,experiment_name)
    with open(f"{save_file}.json", 'w') as json_file:
        json.dump(results_list, json_file)
        
        
          
    total_accuracy = total_correct / total_count if total_count else 0
    no_accuracy = 1- no_correct / no_count if no_count else 0
    yes_accuracy = yes_correct / yes_count if yes_count else 0
    no_proportion = no_count / total_count if total_count else 0
    yes_proportion = yes_count / total_count if total_count else 0
    #no_accuracy = no_correct / total_count if total_count else 0
    #yes_accuracy = yes_correct / total_count if total_count else 0
    
    with open(f"{save_file}_metrics.json", 'w') as json_file:
        json.dump({
            "total_accuracy": total_accuracy,
            "no_accuracy": no_accuracy,
            "yes_accuracy": yes_accuracy,
            "no_proportion": no_proportion,
            "yes_proportion": yes_proportion,
        }, json_file)
    
    
    if verbose:
        print("Total Accuracy: ", total_accuracy)
        print("No Accuracy: ", no_accuracy)
        print("Yes accuracy: ", yes_accuracy)
        
        
    # TODO: Calculate the score
    # {overall_accuracy: float,yes_accuracy: float, no_accuracy: float}
    return total_accuracy, yes_accuracy, no_accuracy, no_proportion, yes_proportion
    
            
# Run the experiment for ToQD without IR
def run_toqd_without_IR_experiment(
    experiment_name="",
    query_data={},
    toqd=None,
    verbose=False,
    results_dir="",
):
    """Run the experiment for ToQD without IR

    Args:
        experiment_name (str, optional): _description_. Defaults to "".
        query_data (dict, optional): _description_. Defaults to {}.
        toqd (_type_, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to False.
        results_dir (str, optional): _description_. Defaults to "".
    """

    results_list = []
    total_count = 0
    total_correct = 0
    total_time = 0
    
    count = 0
    for data in tqdm(query_data):
        
        try:
            question = data['question']
        
            start = time.time()
            ans,qa = toqd.response(
                input=question,
                verbose=False
            )
            end = time.time()
            total_time += end - start
            match_result = match_or_not(ans, data['answer'])
        
            result_data = data.copy()  
            result_data['ans'] = ans  
            result_data['time'] = end - start
            result_data['flag'] = match_result
            #result_data['qa'] = qa[0]['path_set']
            results_list.append(result_data)
        
            total_count += 1
            total_correct += match_result
            
            count += 1
            if count == 20:
                break

        except Exception as e:
            time.sleep(10)
    
    os.makedirs(results_dir, exist_ok=True)
    # Save the retrieval results
    save_file = os.path.join(results_dir,experiment_name)
    with open(f"{save_file}.json", 'w') as json_file:
        json.dump(results_list, json_file)
        
    total_accuracy = total_correct / total_count if total_count else 0
    average_time = total_time / total_count if total_count else 0
    
    with open(f"{save_file}_metrics.json", 'w') as json_file:
        json.dump({
            "total_accuracy": total_accuracy,
            "average_time": average_time,
        }, json_file)
        
    if verbose:
        print("Total Accuracy: ", total_accuracy)
        print("Average Time: ", average_time)
        
    return total_accuracy, average_time


# Run the experiment for ToQD without IR
def run_toqd_with_IR_experiment(
    experiment_name="",
    query_data={},
    toqd=None,
    verbose=False,
    results_dir="",
    top_n=1,
    max_retries=3
):
    """Run the experiment for ToQD without IR

    Args:
        experiment_name (str, optional): _description_. Defaults to "".
        query_data (dict, optional): _description_. Defaults to {}.
        toqd (_type_, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to False.
        results_dir (str, optional): _description_. Defaults to "".
    """

    results_list = []
    total_count = 0
    total_correct = 0
    total_time = 0
    tmp_input_words = 0
    tmp_rounds = 0
    tmp_output_words = 0
    
    count = 0
    for data in tqdm(query_data):
        
        try:
            question = data['question']
        
            start = time.time()
            pred_answer, input_words, output_words, rounds,original_sub_questions_mapping,sub_questions,graph_content,qa_dialog = toqd.response(
                input=question,
                verbose=False,
                top_n=top_n,
                max_retries=max_retries,
            )
            end = time.time()
            total_time += end - start
            match_result = match_or_not(pred_answer, data['answer'])
        
            result_data = data.copy()  
            result_data['pred_answer'] = pred_answer 
            result_data['time'] = end - start
            result_data['flag'] = match_result
            result_data['input_words'] = input_words
            result_data['output_words'] = output_words
            result_data['rounds'] = rounds
            result_data['original_sub_questions'] = original_sub_questions_mapping
            result_data['sub_questions'] = sub_questions
            result_data['graph_content'] = graph_content
            result_data['qa_dialog'] = qa_dialog
                        
            results_list.append(result_data)
        
            total_count += 1
            total_correct += match_result
            tmp_input_words += input_words
            tmp_output_words += output_words
            tmp_rounds += rounds
            
            
            count += 1
            if count == 20:
                break
            
            
            
        except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep(10)
    
    os.makedirs(results_dir, exist_ok=True)
    # Save the retrieval results
    save_file = os.path.join(results_dir,experiment_name)
    with open(f"{save_file}.json", 'w') as json_file:
        json.dump(results_list, json_file)
        
    total_accuracy = total_correct / total_count if total_count else 0
    average_time = total_time / total_count if total_count else 0
    input_words_per_query = tmp_input_words / total_count if total_count else 0
    output_words_per_query = tmp_output_words / total_count if total_count else 0
    rounds_per_query = tmp_rounds / total_count if total_count else 0
    
    with open(f"{save_file}_metrics.json", 'w') as json_file:
        json.dump({
            "total_accuracy": total_accuracy,
            "average_time": average_time,
            "input_words_per_query": input_words_per_query,
            "output_words_per_query": output_words_per_query,
            "rounds_per_query": rounds_per_query,
        }, json_file)
        
    if verbose:
        print("Total Accuracy: ", total_accuracy)
        print("Average Time: ", average_time)
        
    return total_accuracy, average_time, input_words_per_query, output_words_per_query, rounds_per_query

def run_toqd_with_Robustness_experiment(
    experiment_name="",
    query_data={},
    toqd=None,
    verbose=False,
    results_dir="",
    top_n=1,
    max_retries=3,
    is_toqd=False
):
    """Run the experiment for ToQD without IR

    Args:
        experiment_name (str, optional): _description_. Defaults to "".
        query_data (dict, optional): _description_. Defaults to {}.
        toqd (_type_, optional): _description_. Defaults to None.
        verbose (bool, optional): _description_. Defaults to False.
        results_dir (str, optional): _description_. Defaults to "".
    """

    results_list = []
    total_count = 0
    total_correct = 0

    count = 0
    for data in tqdm(query_data):
        
        try:
            question = data['question']
            
            pred_answer = toqd.response(
                input=question,
                verbose=False,
                top_n=top_n,
                max_retries=max_retries,
                is_toqd=is_toqd,
                graph_content=data['graph_content'],
                sub_questions=data['sub_questions']
            )
            
            match_result = match_or_not(pred_answer, data['answer'])
        
            result_data = data.copy()  
            result_data['pred_answer'] = pred_answer 
            result_data['flag'] = match_result
                        
            results_list.append(result_data)
        
            total_count += 1
            total_correct += match_result   
            
            count += 1
            if count == 50:
                break
            
            
            
        except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep(10)
            #raise ValueError
    
    os.makedirs(results_dir, exist_ok=True)
    # Save the retrieval results
    save_file = os.path.join(results_dir,experiment_name)
    with open(f"{save_file}.json", 'w') as json_file:
        json.dump(results_list, json_file)
        
    total_accuracy = total_correct / total_count if total_count else 0
    
    with open(f"{save_file}_metrics.json", 'w') as json_file:
        json.dump({
            "total_accuracy": total_accuracy,
        }, json_file)
        
    if verbose:
        print("Total Accuracy: ", total_accuracy)
        
    return total_accuracy