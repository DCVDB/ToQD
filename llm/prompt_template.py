class ToQPromptTemplate():
    def __init__(self):
        
        self.templates = {
            
            # General template for Topology of Question Composition
            # ----------------------------------------------------------------
            "general" : {
                
                # Generation question template for Topology of Question Composition general
                
                # Generation 这里 如果是 muti-hop试一下将Each sub-question should capture diverse critical context of the original question clearly, concisely and entirely.\n"
                # 去掉可能会影响效果
                'generation' : ( 
                    "Given the original question: \n"
                    #"Given the question: \n"
                    "------------------------------------\n"
                    "{original_question}"
                    "\n------------------------------------\n"
                    #"You are a Teacher."
                    #"Your task is to decompose original question reasoning steps into 2 logically connected sub-questions for helping students reason towards the answers of the original question.\n"
                    "Your task is to decompose original question reasoning steps into 2~4 logically connected sub-questions for helping students reason towards the answers of the original question.\n"
                    "Each sub-question should capture diverse critical context of the original question clearly, concisely and entirely.\n"
                    "These sub-questions should be logically connected and are designed to guide students towards constructing a comprehensive answer through structured reasoning.\n"
                    "Just output the sub-questions like this: '1. ...'\n"
                    #"Your task is to decompose the reasoning steps of question into 2 logically connected sub-questions for helping students reason towards the answers of the original question.\n"              
                    #" Your task is to decompose the original question into a few logically connected sub-questions for helping students reason towards the answers of the original question.\n"
                    #" Your task is to decompose the original question into 2~4 logically connected sub-questions for helping students reason towards the answers of the original question.\n"
                    #" Your task is to decompose the original question into 3 logically connected sub-questions for helping students reason towards the answers of the original question.\n"
                    #" Your task is to decompose the original question into 2 logically connected sub-questions for helping students reason towards the answers of the original question.\n"
                    #" Your task is to decompose the original question into 2 logically connected sub-questions for helping students reason towards the answers of the original question.\n"
                    #" Your task is to decompose the reasoning steps of original question into 2 logically connected sub-questions for helping students reason towards the answers of the original question.\n"
                    #" Your task is to decompose the reasoning steps of original question into 2 logically connected sub-questions for helping students reason towards the answers of the original question.\n"
                    #"Each sub-question should capture diverse critical context of the original question clearly, concisely and entirely.\n"
                    #"Each sub-question should challenge students to infer missing details or apply deeper understanding beyond the explicitly provided information.\n"
                    #"Each sub-question should challenge students to infer missing details or apply deeper understanding beyond the context of explicitly original question.\n" 
                    #"Each sub-question should challenge students to infer missing details or apply deeper understanding beyond the context of original question.\n"
                    #"These sub-questions should be logically connected and are designed to guide students towards constructing a comprehensive answer through structured reasoning."
                ),
                
                # Build the original question and its sub-question topology map for Topology of Question Composition general
                'build' : (
                    #"Given the main question and its sub questions:\n"
                    #"------------------------------------\n"
                    #"MainQuestion:'{original_question}'"
                    #"\n------------------------------------\n"
                    #"SubQuestions:\n{sub_questions}"
                    #"\n------------------------------------\n"
                    #"Your task is to build a topological graph to analyze the reasoning process among the sub-questions based on the main question.\n"
                    #"Represent the main question as 'Q' and just label each sub-question by its respective numbers. \n"
                    #"Use the symbol '->' to denote the hierarchical relationships paths, where the left side question provides useful information for the right side question.\n"
                    #"Clearly depict individual arrows from each sub-question pointing towards the main question or to another sub-question that it helps to answer.\n"
                    #"This detailed topology graph is designed to guide students towards constructing a comprehensive answer through structured reasoning."
                    
                    "Given the main question and its sub-questions:\n"
                    "------------------------------------\n"
                    "Main question:'{original_question}'"
                    "\n\nSub questions:\n{sub_questions}"
                    "\n------------------------------------\n"
                    #"You are a Teacher."
                    #"Given the sub-questions:\n"
                    #"------------------------------------\n"
                    #"{sub_questions}"
                    #"\n------------------------------------\n"
                    #"Your task is to build a topological graph to analyze the reasoning process among the sub-questions based on the original question: '{original_question}'.\n"
                    #"Use the symbol '->' to denote the reasoning paths, where the left side question provides useful information for the right side question.\n"
                    #"Represent the original question as 'Q' and just label each sub-question by its respective numbers. \n"
                    #"Clearly depict individual arrows from each sub-question pointing towards the main question or to another sub-question that it helps to answer.\n"
                    #"This structured topology is designed to guide students in constructing a comprehensive answer through a clearly defined reasoning process."
                    #"Your task is to build a topological graph to analyze the reasoning process among the sub-questions based on the main question.\n"
                    #"Represent the main question as 'Q' and just label each sub-question by its respective numbers. \n"
                    #"Use the symbol '->' to denote the reasoning paths, where the left side question provides useful information that helps answer the right side question.\n"
                    #"Clearly depict individual arrows from each sub-question pointing towards the main question or to another sub-question that it helps to answer.\n"
                    #"This detailed topology graph should are designed to guide students towards constructing a comprehensive answer through structured reasoning."
                    "You are a Teacher."
                    "Your task is to build a topological graph to analyze the reasoning process among the sub-questions and between the sub-questions and the main question.\n"
                    "Represent the main question as 'Q' and just label each sub-question by its respective numbers. \n"
                    "Use the symbol '->' to denote the reasoning paths, where the left side question (e.g., a sub-question) provides useful information that helps answer the right side question (e.g., the main question or another sub-question).\n"
                    "Ensure that all sub-questions are interconnected in a way that they collectively and coherently contribute to answering the main question.\n"
                    "Clearly depict individual arrows from each sub-question pointing towards the main question or to another sub-question that it helps to answer.\n"
                    "This detailed topology graph should are designed to guide students towards constructing a comprehensive answer through structured reasoning."
                ),
                
        
                
                # Summary question-answer dialog record template for Topology of Question Composition general
                # Answer generation for different benchmarks  
                'summary' : (
                    "Given the relevant information: \n"
                    "------------------------------------\n"
                    "{qa_dialog}"
                    "\n------------------------------------\n"
                    #"Please provide a concise and simple answer to the question: '{original_question}'.\n"
                    #"Please provide a concise and simple answer just for one or two words to the question: '{original_question}'.\n"
                    #"Please provide a concise and simple answer of the question: '{original_question}'.\n"
                    #"Please provide a simple answer of the question: '{original_question}'.\n"
                    "Please provide the answer of the question: '{original_question}'.\n"
                ),
                
                # Verify the irrelevant sub-questions for Topology of Question Composition general
                'verification' : (
                    "Given the original question and its sub-questions:\n"
                    "------------------------------------\n"
                    "Original question:'{original_question}'"
                    "\n\nSub questions:\n{sub_questions}"
                    "\n------------------------------------\n"
                    #"You are a Teacher. Your task is:\n"
                    " Your task is:\n"
                    " 1. Filter out the sub-question cannot help students reason towards the answers of the original question.\n"
                    " 2. Filter out the sub-question cannot capture critical context of the original question clearly and concisely.\n"
                    " 3. Filter out the sub-question semantics repetition.\n"
                    #" 3. Filter out any off-topic or irrelevant sub-question.\n"
                    #" 1. Filter out the sub-question cannot capture critical context of the original question clearly and concisely.\n"
                    #" 2. Filter out any off-topic or irrelevant sub-question.\n"
                    "Please label the filtered out sub-question by its number."
                ),
                
                
            },
            
            # Leaf node for Topology of Question Composition
            # ----------------------------------------------------------------
            "leaf_node" : {
                
                # Leaf node verification for Topology of Question Composition
                "verification" : (
                    "Can you answer the following question: '{question}'?\n"
                    #" If not, simply respond with 'No!'; otherwise, please provide the detailed answer to the question '{question}'."
                   # " If you couldn't, simply respond with 'NO!'; otherwise, please provide a simple answer to the question: '{question}'."
                   " If you couldn't, just simple respond with 'NO!'; otherwise, please provide a simple answer to the question: '{question}'."
                   #" If you couldn't answer the question, just simple respond with 'NO!'; otherwise, please provide a simple answer to the question: '{question}'."
                   #" If you couldn't, simply respond with 'NO!'; otherwise, please provide the answer to the question: '{question}'."
                    #"If you could, please provide a simple answer of the '{question}'. If not, simply respond with 'NO!'."
                    #"If you could, please provide the answer of the '{question}'. If not, simply respond with 'NO!'."
                ),
                
                # Leaf node RAG verification for Topology of Question Composition
                "rag_verification" : (
                    "Given the context information: \n"
                    "------------------------------------\n"
                    "{context}"
                    "\n------------------------------------\n"
                    "Can you answer the following question based on the context information: '{question}'?"
                    #" If yes, please provide the answer of question '{question}'. If not, simply respond with 'NO!'."
                    #" If not, simply respond with 'NO!'; otherwise, please provide a simple answer to the question '{question}'."
                    "If you could, please provide the answer of the '{question}'. If not, simply respond with 'NO!'."
                ),
                
                # Question Decomposition for Topology of Question Composition in the leaf node
                "decomposition" : (
                    "Given the original question and context: \n"
                    "------------------------------------\n"
                    "Original question:\n{original_question}"
                    "\n\n"
                    "Context:\n{context}"
                    "\n------------------------------------\n"
                    #"You are a Teacher."
                    " Your task is to identify aspects of the original question that remain unresolved even with the provided context."
                    " Decompose these unresolved aspects int original question into two simple sub-questions that are entirely separated from the context."
                    #" Decompose these unresolved aspects for original question into a few sub-questions that are entirely separated from the context."
                    #" Each sub-question should capture unresolved critical context of the original question clearly, concisely and entirely.\n"
                    " Each sub-question should challenge students to infer missing details or apply deeper understanding beyond the explicitly provided information.\n"
                    #" Each sub-question should challenge students to infer missing details or apply deeper understanding that extends beyond the explicitly original question and context."
                    " These sub-questions should be focused on the unresolved aspects of original question and designed to guide students towards constructing a comprehensive answer through structured reasoning."
                ),
                
                
                # Just answer the question if the topological map is too complex for the leaf node.
                'answer': (
                    "Given the relevant information: \n"
                    "------------------------------------\n"
                    "{context}"
                    "\n------------------------------------\n"
                    #"Please provide a concise and simple answer to the question: '{question}'.\n"
                    "Please provide the answer to the question: '{question}'.\n"
                )
            },
            
            # Internal node for Topology of Question Composition
            # ----------------------------------------------------------------
            "internal_node" : {
                
                # Rewrite the original question for Topology of Question Composition in the internal node
                "rewrite" : (
                   "Given the answers from the sub-questions: \n"
                    "------------------------------------\n"
                    "{qa_dialog}"
                    "------------------------------------\n"
                    #"You are a Teacher."
                    "Your task is to rewrite the main question: '{question}' to just incorporate the answers from the sub-questions directly into the main question.\n"
                    " Avoid repetition of the information already provided in the sub-questions.\n"
                    " The new question should be simply and concisely help students reasoning the original question: '{original_question}'."
                    #"Your task is to rewrite the main question: '{question}' to based on the answers from the sub-questions."
                ),  
                
                # Question Decomposition for Topology of Question Composition in the internal node
                "decomposition" : (
                    "Given the original question and relevant information: \n"
                    "------------------------------------\n"
                    "Original question:\n{original_question}"
                    "\n\n"
                    "Relevant information:\n{context}\n{qa_dialog}"
                    "\n------------------------------------\n"
                    #"You are a Teacher."
                    " Your task is to identify aspects of the original question that remain unresolved even with the relevant information."
                    #" Decompose these unresolved aspects int original question into a few sub-questions that are entirely separated from the relevant information."
                    " Decompose these unresolved aspects int original question into two sub-questions that are entirely separated from the relevant information."
                    #" Decompose these unresolved aspects int original question into two simple sub-questions that are entirely separated from the relevant information."
                    " Each sub-question should challenge students to infer missing details or apply deeper understanding beyond the explicitly provided information.\n"
                    #" Each sub-question should challenge students to infer missing details or apply deeper understanding that extends beyond the explicitly original question and relevant information."
                    " These sub-questions should be focused on the unresolved aspects of original question and designed to guide students towards constructing a comprehensive answer through structured reasoning."
                ), 
                
                
            }
        }
        
    # Load function
    # ----------------------------------------------------------------
    def load(
        self,
        node_type, 
        template_type,
        **kwargs
    ):
        try:
            template = self.templates[node_type][template_type]
            return template.format(**kwargs)
        except Exception as e:
            raise ValueError(f"{node_type} or {template_type} is error : {e}")



class CustomPromptTemplate():
    
    def __init__(self) -> None:
        
        self.templates = {
            
            "general": {
                
                # Directly prompt
                # ---------------------------------------------
                "original" : (
                    "Please provide a simple answer to the question '{question}'."
                ), 
                
                # CoT
                # --------------------------------------------
                "CoT" : (
                    "Please provide a simple answer to the question '{question}'.\n"
                    "Let's think step by step."
                ),
                
                # RAG
                # ------------------------------------------
                "rag" : (
                    "Given the context information: \n"
                    "------------------------------------\n"
                    "{context}"
                    "\n------------------------------------\n"
                    "Please provide a simple answer to the question '{question}'.\n"
                    #"Let's think step by step."
                ),
                
                
                
            },
            
            "self-verify": {
                
                # Directly prompt
                # ---------------------------------------------
                "original": (
                    #"You are a honest student."
                    "Can you answer the following question: '{question}'?"
                    #" If not, simply respond with 'No!'; otherwise, please provide the answer to the question '{question}'."
                    #" If not, respond with 'NO!'; otherwise, please provide a simple answer to the question '{question}'."
                    #" If you can, please provide the answer of '{question}'; otherwise, simply respond with 'NO!'"
                    "If you could, please provide a simple answer of the '{question}'. If not, simply respond with 'NO!'."
                    #"If you could, please answer 'YES!'. If not, simply respond with 'NO!'."
                    #"If not, simply respond with 'NO!'; otherwise, please simply respond with 'YES!'"
                ),
                
                # CoT
                # --------------------------------------------
                "CoT" : (
                    "Let's think step by step."
                    "Can you answer the following question: '{question}'?\n"
                    #"Let's think step by step."
                    #" If not, simply respond with 'NO!'; otherwise, please provide a simple answer to the question '{question}' and think step by step.\n"
                    #"If you could, please provide the answer of '{question}' and think step by step'. If not, simply respond with 'NO!'."
                    #"If you could, please provide the answer of '{question}' and think step by step'. If not, respond with 'NO!' and explain the reason."
                    "If you could, please provide a simple answer of the '{question}'. If not, simply respond with 'NO!'."
                    #"Let's think step by step."
                ),
                
                # RAG
                # --------------------------------------------
                "rag" : (
                    "Given the relevant information: \n"
                    "------------------------------------\n"
                    "{context}"
                    "\n------------------------------------\n"
                    "Can you answer the following question based on the relevant information: '{question}'?"
                    #"Can you answer the following question: '{question}'?\n"
                    #" If yes, please provide the answer of question '{question}'. If not, simply respond with 'NO!'."
                    #" If not, simply respond with 'NO!'; otherwise, please provide a simple answer to the question '{question}'."
                    "If you could, please provide the answer of the '{question}'. If not, simply respond with 'NO!'."
                    #"If you could, please provide the simple answer of the '{question}'. If not, simply respond with 'NO!'."
                ),
            }
            
            
            
        }


    # Load function
    # ----------------------------------------------------------------
    def load(
        self,
        node_type, 
        template_type,
        **kwargs
    ):
        try:
            template = self.templates[node_type][template_type]
            return template.format(**kwargs)
        except Exception as e:
            raise ValueError(f"{node_type} or {template_type} is error : {e}")
































