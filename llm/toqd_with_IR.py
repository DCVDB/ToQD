import logging
import re
import numpy as np

from .prompt_template import ToQPromptTemplate
from .utils import build_adjacency_matrix
from collections import deque
from .normalize_text import normalize

class ToQDWithIR(ToQPromptTemplate):
    def __init__(
        self,
        llm=None,
        retriever=None,
    ) -> None:
        """Topology of Question Composition  

        Args:
            llm (_type_, optional): The large language model used for the answer the question. Defaults to None.
            retriever (_type_, optional): The retriever function for the rag pipeline. Defaults to None.
        """
        
        # Basic configuration
        self.llm = llm
        self.retriever = retriever
        
        super().__init__()
        
    # load the prompt template
    # ----------------------------------------------------------------
    def __load(
        self, 
        node_type, 
        template_type, 
        **kwargs
    ) -> str:
        try:
            return super().load(node_type, template_type, **kwargs)
        except ValueError as e:
            logging.error(f"Error loading template: {e}")
            raise ValueError(f"Error loading template: {e}")
        
        
    
    # Response functions
    # ----------------------------------------------------------------
    def response(
        self,
        input="",
        verbose=False,
        max_retries=3,
        top_n=1
    ) -> str:
        
        """Response the input  

        Args:
            input (str, optional): Input the question. Defaults to "".
            verbose (bool, optional): Whether print the cot process. Defaults to False.
        """
        rounds = 0
        input_words = 0
        output_words = 0
        
        # Generate the sub-questions
        # ----------------------------------------------------------------
        sub_questions, tmp_input_words, tmp_output_words,original_sub_questions_mapping = self.__gen_filter_sub_questions(
            input=input,
            verbose=verbose,
            max_retries=max_retries
        )
        input_words += tmp_input_words
        output_words += tmp_output_words

        # Build topology graph
        # ----------------------------------------------------------------
        graph, tmp_input_words, tmp_output_words,graph_content = self.__build_topology_graph(
            sub_questions=sub_questions,
            original_question=input,
            verbose=verbose,
        )
        input_words += tmp_input_words
        output_words += tmp_output_words
        
        # Add the original question into the sub_questions_mapping
        sub_questions[0] = input
        
        
        # Topology graph sorting
        # ----------------------------------------------------------------
        topology_order, question_node_mapping, tmp_input_words, tmp_output_words, rounds = self.__topology_sort(
            graph=graph,
            sub_questions=sub_questions,
            verbose=verbose,
            top_n=top_n,
        )
        
        # Summary the qa_dialog and answer the questions
        qa_dialog = self.__construct_qa_dialog(
            topology_order=topology_order,
            question_node_mapping=question_node_mapping,
            verbose=verbose
        )
        
        summary_content = self.__load(
            node_type='general',
            template_type='summary',
            qa_dialog=qa_dialog,
            original_question=input
        )
        input_words += self.__count_words(input_string=summary_content)
        
        answer = self.llm.response(
            input=summary_content
        )
        output_words += self.__count_words(input_string=answer)
        
        if verbose:
            print("--------------------------------")
            print("The final answer is:")
            print(answer)
            
        return answer, input_words, output_words, rounds,original_sub_questions_mapping,sub_questions,graph_content,qa_dialog
        
    
    # Generate the sub-questions and filter out the irrelevant questions
    def __gen_filter_sub_questions(
        self, 
        input="", 
        verbose=False,
        max_retries=3
    ) -> dict:
        
        count = 0
        while count < max_retries:
            
            output_words = 0
            input_words = 0
            # M generate the sub-questions
            # --------------------------------
            generation_content = self.__load(
                node_type='general',
                template_type='generation',
                original_question=input,
            )

            input_words += self.__count_words(input_string=generation_content)
            
            sub_questions = self.llm.response(input=generation_content)
            output_words += self.__count_words(input_string=sub_questions)
    
            pattern = r"(\d+)\.\s*(.*?)\s*$"
            matches = re.findall(pattern, sub_questions, re.MULTILINE)
    
            sub_questions_mapping = {int(num): question for num, question in matches}

            # M filter out the irrelevant sub-questions
            # --------------------------------
            filter_content = self.__load(
                node_type='general',
                template_type='verification',
                original_question=input,
                sub_questions=sub_questions
            )
            input_words += self.__count_words(input_string=filter_content)
            
            filter_questions = self.llm.response(input=filter_content)
            output_words += self.__count_words(input_string=filter_questions)
    
            # Parsing filtered questions to keep
            keep_questions = re.findall(r"(\d+)\.", filter_questions)
            keep_questions = set(map(int, keep_questions))
    
            # Remove filtered out questions
            final_questions = {num: q for num, q in sub_questions_mapping.items() if num not in keep_questions}
            renumbered_final_questions = {i + 1: question for i, (_, question) in enumerate(final_questions.items())}
        
            if verbose:
                print("Original Sub-Questions:")
                print(sub_questions)
                print("--------------------------------")
                print(filter_questions)
                print("--------------------------------")
                print("Final Sub-Questions:")
                for num in sorted(renumbered_final_questions):
                    print(f"{num}. {renumbered_final_questions[num]}")
                print("--------------------------------")
                
            if len(renumbered_final_questions) >= 1:
                return renumbered_final_questions,input_words,output_words,sub_questions_mapping
            
            count += 1
            if count == max_retries:
                return renumbered_final_questions,input_words,output_words,sub_questions_mapping
        
    """
    # Generate the sub-questions and filter out the irrelevant questions
    def __gen_filter_sub_questions(
        self, 
        input="", 
        verbose=False,
        max_retries=3
    ) -> dict:
        output_words = 0
        input_words = 0
        # M generate the sub-questions
        # --------------------------------
        generation_content = self.__load(
            node_type='general',
            template_type='generation',
            original_question=input,
        )

        input_words += self.__count_words(input_string=generation_content)
            
        sub_questions = self.llm.response(input=generation_content)
        output_words += self.__count_words(input_string=sub_questions)

        #print(type(sub_questions))
        #sub_questions = normalize(sub_questions)
        #sub_questions = sub_questions.replace('-','')
        #sub_questions = sub_questions.replace(':\n','')
        pattern = r"(\d+)\.\s*(.*?)\s*$"
        matches = re.findall(pattern, sub_questions, re.MULTILINE)

            
        sub_questions_mapping = {int(num): question for num, question in matches}

        if verbose:
            print("Sub-Questions:")
            print(sub_questions)
            print("--------------------------------")
    
        return sub_questions_mapping ,input_words,output_words,sub_questions_mapping
    """   
    
    
    
    
    # Build the topology graph from the sub-questions and original questions
    def __build_topology_graph(
        self,
        sub_questions={}, 
        original_question="",
        verbose=False
    ) -> list:
        
        input_words = 0
        output_words = 0
        
        # Build the string of sub-questions
        sub_questions_content = f""
        for num in sub_questions:
            sub_questions_content+= f"{num}. {sub_questions[num]}\n"
            
        # M predicts the topology graph from the original questions and sub-questions
        content = self.__load(
            node_type="general",
            template_type="build",
            original_question=original_question,
            sub_questions=sub_questions_content
        )
        input_words += self.__count_words(input_string=content)
        
        graph_content = self.llm.response(
            input=content
        )
        
        
        adjacency_matrix = build_adjacency_matrix(graph_content)
        
        if verbose:
            print("Graph content:")
            print("--------------------------------")
            print(graph_content)
            print("Adjacency matrix:")
            print("--------------------------------")
            for row in adjacency_matrix:
                print(row)
            
        return adjacency_matrix,input_words,output_words,graph_content
    
    
    # Self verification function
    # ----------------------------------------------------------------
    def __self_verification(
        self,
        question="",
        state="leaf_node",
        verbose=False,
        top_n=1,
        qa_dialog="",
        original_question=""
    ):
        if state == "leaf_node":
            
            return self.__self_verification_leaf_node(
                    question=question,
                    verbose=verbose,
                    top_n=top_n
                )
            
        elif state == "internal_node":
            
            rewrite_question = self.llm.response(
                input=self.__load(
                    node_type="internal_node",
                    template_type="rewrite",
                    qa_dialog=qa_dialog,
                    question=question,
                    original_question=original_question
               )
            )
            if verbose:
                print("--------------------------------")
                print("Rewritten question")
                print(rewrite_question)
                print("--------------------------------")
                
            ver, tmp_input_words, tmp_output_words, tmp_rounds = self.__self_verification_leaf_node(
                question=rewrite_question,
                verbose=verbose,
                top_n=top_n,
            )
            
            return ver, tmp_input_words, tmp_output_words, tmp_rounds, rewrite_question
        else:
            raise ValueError("The state must be either leaf_node or internal_internal")
            
    # leaf_node 
    def __self_verification_leaf_node(
        self,
        question="",
        verbose=False,
        top_n=1
    ):
        
        input_words = 0
        output_words = 0
        rounds = 0
        
        # M analysis itself answer for the question
        ver_content = self.__load(
            node_type="leaf_node",
            template_type="verification",
            question=question
        )
        input_words += self.__count_words(input_string=ver_content)
        
        ver = self.llm.response(
            input=ver_content
        )
        output_words += self.__count_words(input_string=ver)
        
        # Rag verification
        is_rag = re.match(r'^NO!', ver) is not None
        
        if is_rag:
            context = self.retriever.search_document(question, top_n=top_n)
            rounds += 1
            
            # M analysis itself answer for the question based on the rag pipeline
            rag_context = self.__load(
                node_type="leaf_node",
                template_type="answer",
                context=context,
                question=question
            )
            input_words += self.__count_words(input_string=rag_context)
            
            ans = self.llm.response(
                input=self.__load(
                    node_type="leaf_node",
                    template_type="answer",
                    context=context,
                    question=question
                )
            )
            output_words += self.__count_words(input_string=ans)
            
            
            if verbose:
                print("Self node verification.")
                print("--------------------------------")
                print(ver)
                print("--------------------------------")
                print("RAG")
                print("--------------------------------")
                print("Context")
                print(context)
                print("ans")
                print(ans)
                print("--------------------------------")
                
            return ans, input_words, output_words, rounds
 
        else:
            
            if verbose:
                print("Self node verification.")
                print("--------------------------------")
                print(ver)
                print("--------------------------------")
                
            return ver, input_words, output_words, rounds
        
    
    
    # Topology sort
    # ----------------------------------------------------------------
    def __topology_sort(
        self,
        graph,
        sub_questions={},
        verbose=False,
        top_n=1,
    ) -> list[int]:

        """Topological sort using Kahn's algorithm

        Returns:
            _type_: _description_
        """
        
        # Basic configuration
        # ----------------------------------------------------------------
        num_nodes = len(graph)
        in_degree = [0] * num_nodes
        out_degree = [0] * num_nodes  # to track outgoing edges
        question_node_mapping = {}
        
        # This function helps to calculate the number of words
        input_words = 0
        output_words = 0
        rounds = 0
        
        # Initialize node properties with question and an empty set for comprehensive path data
        for i in sub_questions.keys():
            question_node_mapping[i] = {
                "question": sub_questions[i],
                "answer": "",  # Placeholder for an answer
                "path_set": set(),  # Set to store all unique paths' data leading to this node
                "node_type": "undefined"  # Will update to 'internal_node' or 'leaf_node'
            }
            
        # Compute in-degree of each node by checking adjacency matrix
        for i in range(num_nodes):
            for j in range(num_nodes):
                if graph[j][i] == 1:
                    out_degree[i] += 1
                    in_degree[i] += 1
                    
        # Update node type based on out-degree
        for i in sub_questions.keys():
            if out_degree[i] > 0:
                question_node_mapping[i]["node_type"] = "internal_node"
            else:
                question_node_mapping[i]["node_type"] = "leaf_node"
                    
        # Initialize the queue with nodes that have in-degree 0 and are listed in questions
        queue = deque([i for i in range(num_nodes) if in_degree[i] == 0 and i in sub_questions])
        topology_order = []
        
        # Traverse the topology graph
        # ----------------------------------------------------------------
        while len(queue) > 1 or (len(queue) == 1 and queue[0] != 0):
            
            # Pop the question node from the queue
            node = queue.popleft()
            node_state = question_node_mapping[node]["node_type"]
            
            # M self-verification the question node
            # -----------------------------------------------------
            
            # Leaf_node
            if node_state == "leaf_node":
                
                # Self verification
                ver, tmp_input_words, tmp_output_words, tmp_rounds = self.__self_verification(
                    question=question_node_mapping[node]["question"],
                    state="leaf_node",
                    verbose=verbose,
                    top_n=top_n
                )
                
                rounds += tmp_rounds
                input_words += tmp_input_words
                output_words += tmp_output_words
                
            elif node_state == "internal_node":  # Internal node
                
                qa_dialog = ""
                for path in question_node_mapping[node]['path_set']:
                    qa_dialog += f"Q: {path[1]}\nA: {path[2]}\n"
                    
                if verbose:
                    print(qa_dialog)
                
                # Self verification
                ver, tmp_input_words, tmp_output_words, tmp_rounds, rewrite_question = self.__self_verification(
                    question=question_node_mapping[node]["question"],
                    state=node_state,
                    top_n=top_n,
                    verbose=verbose,
                    qa_dialog=qa_dialog,
                    original_question=question_node_mapping[0]["question"]
                )
                
                rounds += tmp_rounds
                input_words += tmp_input_words
                output_words += tmp_output_words
                #question_node_mapping[node]["question"] = rewrite_question
                #current_path_data = (node, rewrite_question, ver)  # 更新current_path_data以使用rewrite_question
                
            question_node_mapping[node]['answer'] = ver
            current_path_data = (node, sub_questions[node], ver)
            #current_path_data = (node, question_node_mapping[node]["question"], ver)
            topology_order.append(node)
            
            # Update the node state
            for adj in range(num_nodes):
                if graph[node][adj] == 1:
                        
                    # Include all predecessors' path data plus current node's data
                    new_path_set = question_node_mapping[node]['path_set'].copy()
                    new_path_set.add(current_path_data)
                    question_node_mapping[adj]['path_set'].update(new_path_set)
                    in_degree[adj] -= 1
                    if in_degree[adj] == 0:
                        queue.append(adj)
        if len(topology_order) == len(sub_questions) - 1:
            return topology_order, question_node_mapping, input_words,output_words,rounds
        else:
            return "Cycle detected, no topological sort possible.", question_node_mapping
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Helper function
    # ------------------------------------------------
    def __construct_qa_dialog(self, topology_order, question_node_mapping, verbose=False):
    
        node_zero_paths = question_node_mapping[0]['path_set']
        path_dict = {path[0]: (path[1], path[2]) for path in node_zero_paths}

        qa_dialog = ""

        for node in topology_order:
            if node in path_dict:  
                question, answer = path_dict[node]
                qa_dialog += f"Q: {question}\nA: {answer}\n"
                
        if verbose:
            print("Qa dialog:")
            print("--------------------------------")
            
            print(qa_dialog)

        return qa_dialog
    
    def __count_words(self,input_string):
        # 使用split方法将字符串按空格分割成单词列表
        
        normalize_input_string = normalize(input_string)
        words = normalize_input_string.split()
        return len(words)