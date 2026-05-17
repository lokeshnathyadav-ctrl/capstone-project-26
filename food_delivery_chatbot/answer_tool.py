import json          # to objectify Java script notion
import os            # to perform operating system related functions
import pandas as pd  # manipulating and working with data
import spacy
# Supress unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Defining a function to refine raw responses into precise, polished answers to users
class Answering_Tool:
    def init(self, language_model = "en_core_web_sm"):
        """
        Initialize the Answering_Tool with a SpaCy language model.

        Args:
        - language_model(str): The SpaCy language model to use (default:"en_core_web_sm")
        """
        self.nlp=spacy.load(language_model)

    def generate_answer(self,question,raw_response):
        """
        Generate a user-friendly answer based on a question and a raw response.

        Args:
        - question (str): The question being asked
        - raw_response (str): The raw response to polish

        Returns
        - str: The polished answer
        """
        # Process the question and raw response with Spacy
        question_doc = self.nlp(question)
        raw_response_doc = self.nlp(raw_response)

        # Perform answer generation tasks, such as:
        # - Identifying relevant entities
        # - Determining the answer's sentiment
        polished_answer = self.polish_answer(raw_response)

        return polished_answer
answer_tool = Tool(
    name = "PolishedResponses",
    func = Answering_Tool,
    description = "Modifies the raw responses obtained from order query tool into polished user-friendly responses.")
