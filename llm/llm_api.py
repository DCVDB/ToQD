from llama_index.llms.openai import OpenAI

from langchain_community.llms import QianfanLLMEndpoint
from llama_index.core.llms import ChatMessage
import logging
import requests
import json
import qianfan
from transformers import AutoTokenizer, AutoModelForCausalLM

# LLM
# ----------------------------------------------------------------
class LLMInterface:
    def __init__(
        self, 
        model_name="", 
        api_key="", 
        secret_key=None, 
        request_timeout=30, 
        max_retries=3, 
        temperature=0.95, 
        top_k=30,
        top_p=0.8, 
        streaming=False,
        repo="./repo",
        platform="openai",
        api_token="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"
    ):
        """The LLM interface for the ToQC

        Args:
            model_name (str, optional): The model name of the LLMs. Defaults to "".
            api_key (str, optional): The api key for openai or baidu. Defaults to "".
            secret_key (_type_, optional): The secret key for qianfan api. Defaults to None.
            request_timeout (int, optional): _description_. Defaults to 30.
            max_retries (int, optional): _description_. Defaults to 3.
            temperature (float, optional): _description_. Defaults to 0.95.
            top_p (float, optional): _description_. Defaults to 0.8.
            streaming (bool, optional): _description_. Defaults to False.
            repo (str, optional): _description_. Defaults to "./repo".
            platform (str, optional): 'openai', 'baidu', 'huggingface','modelscope' Defaults to "openai".
        """
        
        assert platform in ['openai', 'baidu', 'huggingface','modelscope'],  f"[ERROR] the {platform} should be in the ['openai', 'baidu', 'huggingface','modelscope']"
        
        # Basic configuration
        self.api_key = api_key
        self.secret_key = secret_key
        self.model_name = model_name
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.streaming = streaming
        self.repo = repo
        self.platform = platform
        self.api_token = api_token
        
        # init_llm
        self.llm = self.__init_llm()

        
        
    def __init_llm(self):
        if self.platform == "openai":
            return self.__init_openai_llm()
        elif self.platform  == "baidu":
            return self.__init_baidu_llm()
        elif self.platform == "huggingface":
            return self.__init_HF_llm()
        else:
            return self.__init_modelscope_llm()
            
    def __init_openai_llm(self):
        """Set the openai llm instance"""
        
        client = OpenAI(
            model=self.model_name,
            api_key=self.api_key,
            timeout=self.request_timeout,
            max_retries=self.max_retries,
            temperature=self.temperature,
            streaming=self.streaming,
        )
        
        client.additional_kwargs = {
            "top_p" : self.top_p
        }
                
        logging.info(f"[INFO] Set the llm - {self.model_name} successfully")
        
        return client
            
    def __init_baidu_llm(self):
        
        """
        llm = QianfanLLMEndpoint(
                model = self.model_name,
                #streaming=self.streaming,
                #request_timeout = self.request_timeout,
                #temperature = self.temperature,
                #top_p = self.top_p,
                #qianfan_ak = self.api_key,
                #qianfan_sk = self.secret_key,
            )
        """
        
        llm = qianfan.ChatCompletion()
        
        #llm = None
        return llm
         
         
            
            
            
            
            
            
            
    def __init_HF_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.repo, use_auth_token=self.api_token)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.repo, use_auth_token=self.api_token, device_map="auto")
        
        self.tokenizer = tokenizer
        return model
        
        
    def __init_modelscope_llm(self):
        print("Hello, modelscope")
        
        
    # Query functions
    # ----------------------------------------------------------------
    def response(self,input):
        if self.platform == "openai":
            return self.__response_openai_llm(input=input)
        elif self.platform  == "baidu":
            return self.__response_baidu_llm(input=input)
        elif self.platform == "huggingface":
            return self.__response_HF_llm(input=input)
        else:
            return self.__response_modelscope_llm(input=input)
        
    # openai 
    def __response_openai_llm(self,input):
        
        response = self.llm.chat(
                messages = [
                    ChatMessage(
                        role="system", content=input
                    ),
                ]
            )
        
        return response.message.content
    
    # baidu
    def __response_baidu_llm(self,input):
        
        """
        def get_access_token():
            使用 AK，SK 生成鉴权签名（Access Token）
            :return: access_token，或是None(如果错误)
            url = "https://aip.baidubce.com/oauth/2.0/token"
            params = {"grant_type":  "client_credentials", "client_id": self.api_key, "client_secret": self.secret_key}
            return str(requests.post(url, params=params).json().get("access_token"))
        
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_2_7b?access_token=" + get_access_token()
    
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": f"{input}"
                },
            ]
        })
        
        headers = {
            'Content-Type': 'application/json'
        }
    
        response = requests.request("POST", url, headers=headers, data=payload)
        response_dict = json.loads(response.text)
        result_content = response_dict["result"]
        """
        #print(response.text)
        resp = self.llm.do(model=self.model_name,temperature=self.temperature,top_p=self.top_p,messages=[{
            "role": "user",
            "content": input
        }])
        
        #return result_content
        return resp["body"]['result']
    
    
    def __response_HF_llm(self,input):
        
        input_ids = self.tokenizer(input, return_tensors="pt").to("cuda")
        
        outputs = self.llm.generate(
            **input_ids, 
            #max_length=1024,  # Increase the maximum length
            num_beams=5,     # Use beam search for better coherence
            no_repeat_ngram_size=2,  # Avoid repeating n-grams
            early_stopping=True, # Stop early when model converges
            max_new_tokens=256  # Generate up to 100 new tokens
        )
        
        return self.tokenizer.decode(outputs[0])
        



    
    
