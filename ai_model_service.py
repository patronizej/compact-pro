import os
import torch
from fastapi import HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from helpers.sql import clean_generated_sql, generate_sql_prompt, extract_sql_query

from config.settings import conf


class AIModelService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load_model(self):
        """
        Loads the model into memory if not already loaded.
        """
        if not self.is_loaded:
            if torch.cuda.is_available():
                try:
                    self.model, self.tokenizer = self.setup_model(self.model_name)
                    self.is_loaded = True
                except Exception as e:
                    raise RuntimeError("Error loading model: " + str(e))
            else:
                raise Exception("GPU is not available. AI requires GPU to process the request.")

    def unload_model(self):
        """
        Offloads the model to free up GPU memory.
        """
        try:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
        finally:
            self.is_loaded = False

    def setup_model(self, model_name: str):
        """
        Initialize model and tokenizer for AI usage (if available).
        """
        if not os.path.isdir(model_name) or not os.listdir(model_name):
            raise HTTPException(
                status_code=500,
                detail=f"Model directory '{model_name}' does not exist or is empty."
            )

        bnb_config = BitsAndBytesConfig(
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    # ------------------------ SQL GENERATION -------------------------------

    def run_sql_command(self, question: str):
        self.load_model()

        try:
            prompt = generate_sql_prompt(question, "config/prompts/sql_prompt.md")
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=conf.AI_SQL_MAX_TOKENS,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                return_full_text=False,
                num_beams=conf.AI_SQL_NUM_BEAMS,
            )

            # Generate SQL query based on the prompt
            generated_text = pipe(prompt, num_return_sequences=1)[0]["generated_text"]
            sql_query = extract_sql_query(generated_text)

            # Additional level of data protection - clean up the generated SQL query
            return clean_generated_sql(sql_query)

        finally:
            self.unload_model()

    # ------------------------ DATA SUMMARY -------------------------------

    def generate_summary_prompt(
            self,
            sql_question: str,
            data_question: str,
            data: dict,
            prompt_file: str,
    ):
        with open(prompt_file, "r") as prompt_f:
            prompt = prompt_f.read().format(
                user_question=data_question,
                data=data,
            )
            return [
                {"role": "system", "content": prompt},
                {"role": "system", "content": f"Data represent result of SQL query based on user request '${sql_question}'"},
                {"role": "user", "content": data_question},
            ]

    def run_data_question_command(self, sql_question: str, data_question: str, data: dict):
        self.load_model()

        try:
            prompt_messages = self.generate_summary_prompt(
                sql_question,
                data_question,
                data,
                "config/prompts/summary_prompt.md",
            )

            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto",
                return_full_text=False,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_beams=conf.AI_SUMMARY_NUM_BEAMS,
            )

            outputs = pipe(
                prompt_messages,
                max_new_tokens=conf.AI_SUMMARY_MAX_TOKENS,
            )

            return outputs[0]["generated_text"]

        finally:
            self.unload_model()
