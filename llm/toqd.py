import logging
import re
import numpy as np

from .prompt_template import ToQPromptTemplate
from .utils import build_adjacency_matrix
from collections import deque

class ToQD(ToQPromptTemplate):
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
        
        # Generate the sub-questions
        # ----------------------------------------------------------------
        sub_questions = self.__gen_filter_sub_questions(
            input=input,
            verbose=verbose,
            max_retries=max_retries
        )

        # Build topology graph
        # ----------------------------------------------------------------
        graph = self.__build_topology_graph(
            sub_questions=sub_questions,
            original_question=input,
            verbose=verbose,
        )
        
        # Add the original question into the sub_questions_mapping
        sub_questions[0] = input
        
        # Topology graph sorting
        # ----------------------------------------------------------------
        topology_order, question_node_mapping = self.__topology_sort(
            graph=graph,
            sub_questions=sub_questions,
            verbose=verbose,
            top_n=top_n,
            max_retires=max_retries
        )
        
        qa_dialog = self.__construct_qa_dialog(
            topology_order=topology_order,
            question_node_mapping=question_node_mapping,
            verbose=verbose
        )
        
        answer = self.llm.response(
            input=self.__load(
                node_type='general',
                template_type='summary',
                qa_dialog=qa_dialog,
                original_question=input
            )
        )
        
        if verbose:
            print("--------------------------------")
            print("The final answer is:")
            print(answer)
            
        return answer
        
    # Generate the sub-questions and filter out the irrelevant questions
    def __gen_filter_sub_questions(
        self, 
        input="", 
        verbose=False,
        max_retries=3
    ) -> dict:
        
        count = 0
        while count < max_retries:
            # M generate the sub-questions
            # --------------------------------
            generation_content = self.__load(
                node_type='general',
                template_type='generation',
                original_question=input,
            )
    
            sub_questions = self.llm.response(input=generation_content)
    
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
    
            filter_questions = self.llm.response(input=filter_content)
    
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
                
            if len(renumbered_final_questions) >= 2:
                return renumbered_final_questions
            
            count += 1
            if count == max_retries:
                return renumbered_final_questions
    
    
    
    # Build the topology graph from the sub-questions and original questions
    def __build_topology_graph(
        self,
        sub_questions={}, 
        original_question="",
        verbose=False
    ) -> list:
        
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
            
        return adjacency_matrix
            
            
            
            
    # Self verification function
    # ----------------------------------------------------------------
    def __self_verification(
        self,
        question="",
        state="leaf_node",
        verbose=False,
        top_n=1,
        qa_dialog=""
    ):
        """Verification function for the llm response

        Args:
            questions (str, optional): The input question. Defaults to "".
            state (str, optional): The node state of topology . Defaults to "leaf_node" or "internal_node".
        """
        
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
                    question=question
               )
            )

            if verbose:
                print("--------------------------------")
                print("Rewritten question")
                print(rewrite_question)
                print("--------------------------------")
            
            return self.__self_verification_leaf_node(
                    question=rewrite_question,
                    verbose=verbose,
                    top_n=top_n,
                )
        else:
            raise ValueError("The state must be either leaf_node or internal_internal")
        
    # leaf_node 
    def __self_verification_leaf_node(
        self,
        question="",
        verbose=False,
        top_n=1
    ):
        decomposition = False
        
        # M analysis itself answer for the question
        ver = self.llm.response(
            input=self.__load(
                node_type="leaf_node",
                template_type="verification",
                question=question
            )
        )
        
        # Rag verification
        is_rag = re.match(r'^NO!', ver) is not None
        
        if is_rag:
            
            context = self.retriever.search_documents(question, top_n=top_n)
            
            # M analysis itself answer for the question based on the rag pipeline
            rag_ver = self.llm.response(
                input=self.__load(
                    node_type="leaf_node",
                    template_type="rag_verification",
                    context=context,
                    question=question
                )
            )
            
            if verbose:
                print("Self node verification.")
                print("--------------------------------")
                print(ver)
                print("--------------------------------")
                print("RAG verification")
                print("--------------------------------")
                print("Context")
                print(context)
                print()
                print("RAG verification")
                print(rag_ver)
            
            # Harder question need more decomposition
            decomposition = re.match(r'^NO!', rag_ver) is not None
            
            if decomposition:
                return decomposition, context
            else:
                return decomposition, rag_ver
            
        else:
            
            if verbose:
                print("Self node verification.")
                print("--------------------------------")
                print(ver)
                print("--------------------------------")
                
            return decomposition, ver
        
        
    
    # Decomposition the question
    # ----------------------------------------------------------------
    def __self_question_decomposition_generation(
        self,
        question="",
        state="leaf_node",
        verbose=False,
        qa_dialog="",
        context="",
        max_retries=3
    ):
        """_summary_

        Args:
            question (str, optional): _description_. Defaults to "".
            state (str, optional): _description_. Defaults to "leaf_node".
            verbose (bool, optional): _description_. Defaults to False.
            top_n (int, optional): _description_. Defaults to 1.
            qa_dialog (str, optional): _description_. Defaults to "".
            context (str, optional): _description_. Defaults to "".
        """
        
        count = 0
        while count < max_retries:
            
            if state == "leaf_node":
                
                # M decomposition the question
                sub_questions = self.llm.response(
                    input= self.__load(
                        node_type='leaf_node',
                        template_type='decomposition',
                        original_question=question,
                        context=context
                    )
                )
                
            elif state == "internal_node":
                
                # M decomposition the question
                sub_questions = self.llm.response(
                    input= self.__load(
                        node_type='internal_node',
                        template_type='decomposition',
                        original_question=question,
                        context=context,
                        qa_dialog=qa_dialog
                    )
                )
                
        
            pattern = r"(\d+)\.\s*(.*?)\s*$"
            matches = re.findall(pattern, sub_questions, re.MULTILINE)
        
            sub_questions_mapping = {int(num): question for num, question in matches}
        
            filter_questions = self.llm.response(
                input=self.__load(
                    node_type='general',
                    template_type='verification',
                    original_question=question,
                    sub_questions=sub_questions
                )
            )
        
            # Parsing filtered questions to keep
            keep_questions = re.findall(r"(\d+)\.", filter_questions)
            keep_questions = set(map(int, keep_questions))
    
            # Remove filtered out questions
            final_questions = {num: q for num, q in sub_questions_mapping.items() if num not in keep_questions}
            renumbered_final_questions = {i + 1: question for i, (_, question) in enumerate(final_questions.items())}

            if verbose:
                print("--------------------------------")
                print("Decomposition question")
                print(sub_questions_mapping)
                print()
                print(renumbered_final_questions)
                print("--------------------------------")
            
        
            if len(renumbered_final_questions) >= 1:
                return renumbered_final_questions
            
            count += 1
            if count == max_retries:
                return renumbered_final_questions
            
    # Decompose the questions
    # ----------------------------------------------------------------
    def __self_question_decomposition(
        self,
        question="",
        state="leaf_node",
        verbose=False,
        qa_dialog="",
        context="",
        max_retries=3,
        top_n=1
    ) -> str:
        
        attempt_count = 0
        while attempt_count < max_retries:
            
            # Question decomposition generation
            sub_questions = self.__self_question_decomposition_generation(
                question=question,
                state=state,
                verbose=verbose,
                qa_dialog=qa_dialog,
                context=context,
                max_retries=max_retries,
            )
        
            # Check the sub_questions number
            if len(sub_questions) == 1:
                
                # Self-verification
                decomposition, ver = self.__self_verification(
                    question=sub_questions[1],
                    state="leaf_node",
                    verbose=verbose,
                    top_n=top_n,
                )
            

                if decomposition:
                    attempt_count += 1
                    continue
                else:
                    break
            else:
                
                break
            
        if len(sub_questions) == 1 and attempt_count == max_retries:
            
            # Directly answer by the rag pipeline
            sub_questions_context = self.retriever.search_documents(
                sub_questions[1], 
                top_n=top_n
            )
            
            answer = self.llm.response(
                input=self.__load(
                    node_type="leaf_node",
                    template_type="answer",
                    context=sub_questions_context,
                    question=sub_questions[1]
                )
            )
            
            return f"Q:{sub_questions[1]}\nA:{answer}"
            
        elif len(sub_questions) == 1 and attempt_count < max_retries:
            
            return f"Q:{sub_questions[1]}\nA:{ver}"
        
        else:
            
            # Multiple questions need to build the question graph
            graph = self.__build_topology_graph(
                sub_questions=sub_questions,
                original_question=input,
                verbose=verbose,
            )
            
            sub_questions[0] = question

            # Topology sort for children graph
            topology_order, question_node_mapping = self.__topology_sort_for_child_graph(
                graph=graph,
                sub_questions=sub_questions,
                verbose=verbose,
                top_n=top_n,
                max_retires=max_retries
            )
            
            qa_dialog = self.__construct_qa_dialog(
                topology_order=topology_order,
                question_node_mapping=question_node_mapping,
                verbose=verbose
            )
            
            return qa_dialog
        

        
    
    
    
            
            
    # Topology sort
    # ----------------------------------------------------------------
    # Set it as the inner function
    # TODO: Add the functions
    def __topology_sort(
        self,
        graph,
        sub_questions={},
        verbose=False,
        top_n=1,
        max_retires=3
    ) -> list[int]:
        """_summary_

        Args:
            graph (_type_): _description_
            sub_questions (dict, optional): _description_. Defaults to {}.

        Returns:
            list[int]: The node question number
        """
        
        # Basic configuration
        # ----------------------------------------------------------------
        num_nodes = len(graph)
        in_degree = [0] * num_nodes
        out_degree = [0] * num_nodes  # to track outgoing edges
        question_node_mapping = {}
        
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
            
            node = queue.popleft()
            node_state = question_node_mapping[node]["node_type"]
            
            # M self-verification the question node
            # -----------------------------------------------------
            
            # Leaf_node
            if node_state == "leaf_node":
                
                # Self verification
                decomposition, ver = self.__self_verification(
                    question=question_node_mapping[node]["question"],
                    state=node_state,
                    top_n=top_n,
                    verbose=verbose
                )
                
            elif node_state == "internal_node":  # Internal node
                
                qa_dialog = ""
                for path in question_node_mapping[node]['path_set']:
                    qa_dialog += f"Q: {path[1]}\nA: {path[2]}"
                    
                if verbose:
                    print(qa_dialog)
                
                # Self verification
                decomposition, ver = self.__self_verification(
                    question=question_node_mapping[node]["question"],
                    state=node_state,
                    top_n=top_n,
                    verbose=verbose
                )
                
            if decomposition:
                
                # Question decomposition
                sub_questions_qa_dialog = self.__self_question_decomposition(
                    question=question_node_mapping[node]["question"],
                    state=node_state,
                    verbose=verbose,
                    qa_dialog=qa_dialog,
                    context=ver,
                    max_retries=max_retires,
                    top_n=top_n
                )
                
                if node_state == "leaf_node":
                    # Answer decomposition question
                    answer = self.llm.response(
                        input=self.__load(
                            node_type="leaf_node",
                            template_type="answer",
                            context=sub_questions_qa_dialog,
                            question=question_node_mapping[node]["question"]
                        )
                    )
                    
                elif node_state == "internal_node":
                   
                    answer = self.llm.response(
                        input=self.__load(
                            node_type="leaf_node",
                            template_type="answer",
                            context= f"{qa_dialog}\n{sub_questions_qa_dialog}",
                            question=question_node_mapping[node]["question"]
                        )
                    )
                    
                # Answer is ready
                question_node_mapping[node]['answer'] = answer
                current_path_data = (node, sub_questions[node], answer)
                topology_order.append(node)
                    
            else:
                # Answer is ready
                question_node_mapping[node]['answer'] = ver
                current_path_data = (node, sub_questions[node], ver)
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
            return topology_order, question_node_mapping
        else:
            return "Cycle detected, no topological sort possible.", question_node_mapping
    
    
    # Topological sorting for children graph
    # ----------------------------------------------------------------
     # TODO: Add the functions
    def __topology_sort_for_child_graph(
        self,
        graph,
        sub_questions={},
        verbose=False,
        top_n=1,
    ) -> list[int]:
        """_summary_

        Args:
            graph (_type_): _description_
            sub_questions (dict, optional): _description_. Defaults to {}.

        Returns:
            list[int]: The node question number
        """
        
        # Basic configuration
        # ----------------------------------------------------------------
        num_nodes = len(graph)
        in_degree = [0] * num_nodes
        out_degree = [0] * num_nodes  # to track outgoing edges
        question_node_mapping = {}
        
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
            
            node = queue.popleft()
            node_state = question_node_mapping[node]["node_type"]
            
            # M self-verification the question node
            # -----------------------------------------------------
            
            # Leaf_node
            if node_state == "leaf_node":
                
                # Self verification
                decomposition, ver = self.__self_verification(
                    question=question_node_mapping[node]["question"],
                    state=node_state,
                    top_n=top_n,
                    verbose=verbose
                )
                
            elif node_state == "internal_node":  # Internal node
                
                qa_dialog = ""
                for path in question_node_mapping[node]['path_set']:
                    qa_dialog += f"Q: {path[1]}\nA: {path[2]}"
                    
                if verbose:
                    print(qa_dialog)
                
                # Self verification
                decomposition, ver = self.__self_verification(
                    question=question_node_mapping[node]["question"],
                    state=node_state,
                    top_n=top_n,
                    verbose=verbose
                )
                
            if decomposition:
                
                 # Directly answer by the rag pipeline
                sub_questions_context = self.retriever.search_document(
                    question_node_mapping[node]["question"], 
                    top_n=top_n
                )
            
                answer = self.llm.response(
                    input=self.__load(
                        node_type="leaf_node",
                        template_type="answer",
                        context=sub_questions_context,
                        question=question_node_mapping[node]["question"]
                    )
                )
                
                # Answer is ready
                question_node_mapping[node]['answer'] = answer
                current_path_data = (node, sub_questions[node], answer)
                topology_order.append(node)
                
                
            else:
                # Answer is ready
                question_node_mapping[node]['answer'] = ver
                current_path_data = (node, sub_questions[node], ver)
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
            return topology_order, question_node_mapping
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


class ToQDWithoutIR(ToQD):
    
    def __init__(
        self,
        llm=None
    )->None:
        """
        Initialize ToQD without IR
        """
        
        super().__init__(llm=llm)
        
    # Load the LLM prompt template
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
    
    
    # Generate the sub-questions and filter out the irrelevant questions
    def __gen_filter_sub_questions(
        self, 
        input="", 
        verbose=False,
        max_retries=3
    ) -> dict:
        
        count = 0
        while count < max_retries:
            # M generate the sub-questions
            # --------------------------------
            generation_content = self.__load(
                node_type='general',
                template_type='generation',
                original_question=input,
            )
    
            sub_questions = self.llm.response(input=generation_content)
    
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
    
            filter_questions = self.llm.response(input=filter_content)
    
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
                
            if len(renumbered_final_questions) >= 2:
                return renumbered_final_questions
            
            count += 1
            if count == max_retries:
                return renumbered_final_questions
    
    
    
    # Build the topology graph from the sub-questions and original questions
    def __build_topology_graph(
        self,
        sub_questions={}, 
        original_question="",
        verbose=False
    ) -> list:
        
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
            
        return adjacency_matrix
    
    
    
    
    def response(
        self,
        input="",
        verbose=False,
        max_retries=3,
    ) -> str:
        
        """Response the input  

        Args:
            input (str, optional): Input the question. Defaults to "".
            verbose (bool, optional): Whether print the cot process. Defaults to False.
        """
        
        # Generate the sub-questions
        # ----------------------------------------------------------------
        sub_questions = self.__gen_filter_sub_questions(
            input=input,
            verbose=verbose,
            max_retries=max_retries
        )

        # Build topology graph
        # ----------------------------------------------------------------
        graph = self.__build_topology_graph(
            sub_questions=sub_questions,
            original_question=input,
            verbose=verbose,
        )
        
        # Add the original question into the sub_questions_mapping
        sub_questions[0] = input
    
        # Topology graph sorting
        # ----------------------------------------------------------------
        topology_order, question_node_mapping = self.__topology_sort_without_IR(
            graph=graph,
            sub_questions=sub_questions,
            verbose=verbose,
        )
        
        qa_dialog = self.__construct_qa_dialog(
            topology_order=topology_order,
            question_node_mapping=question_node_mapping,
            verbose=verbose
        )
        
        answer = self.llm.response(
            input=self.__load(
                node_type='general',
                template_type='summary',
                qa_dialog=qa_dialog,
                original_question=input
            )
        )
        
        if verbose:
            print("--------------------------------")
            print("The final answer is:")
            print(answer)
            
        return answer,question_node_mapping
        
    def __topology_sort_without_IR(
        self,
        graph,
        sub_questions={},
        verbose=False,
    ) -> list[int]:
    
        """_summary_

        Args:
            graph (_type_): _description_
            sub_questions (dict, optional): _description_. Defaults to {}.

        Returns:
            list[int]: The node question number
        """
        
        # Basic configuration
        # ----------------------------------------------------------------
        num_nodes = len(graph)
        in_degree = [0] * num_nodes
        out_degree = [0] * num_nodes  # to track outgoing edges
        question_node_mapping = {}
        
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
            
            node = queue.popleft()
            node_state = question_node_mapping[node]["node_type"]
            
            
            # Leaf_node
            if node_state == "leaf_node":
                
                question=question_node_mapping[node]["question"]
                
                answer = self.llm.response(
                    input=f"Please provide a simple answer of the '{question}'."
                )

            elif node_state == "internal_node":  # Internal node
                
                qa_dialog = ""
                for path in question_node_mapping[node]['path_set']:
                    qa_dialog += f"Q: {path[1]}\nA: {path[2]}\n"
                    
                if verbose:
                    print(qa_dialog)
                
                question=question_node_mapping[node]["question"]
                
                rewrite_question = self.llm.response(
                    input=self.__load(
                        node_type="internal_node",
                        template_type="rewrite",
                        qa_dialog=qa_dialog,
                        question=question,
                        original_question=sub_questions[0]
                    )
                )
                
                print(rewrite_question)
                tmp_prompt_template = (
                    #"Given the context information: \n"
                    "Given the relevant information: \n"
                    "------------------------------------\n"
                    "{qa_dialog}"
                    "\n------------------------------------\n"
                    #"Please provide a simple answer to the question '{question}'.\n"
                    "Please provide the answer to the question '{question}'.\n"
                )
                
                answer = self.llm.response(
                    input=tmp_prompt_template.format(qa_dialog=qa_dialog,question=rewrite_question)
                )
                
                question_node_mapping[node]["question"]=rewrite_question
                
                
            question_node_mapping[node]['answer'] = answer
            current_path_data = (node, sub_questions[node], answer)
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
            return topology_order, question_node_mapping
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