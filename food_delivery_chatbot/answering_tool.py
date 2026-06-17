import spacy
# Defining a function to refine raw responses into precise, polished answers to users
class Answering_Tool:
    def init(self, language_model = "en_core_web_sm"):
        """
        Initialize the Answering_Tool with a SpaCy language model.

        Args:
        - language_model(str): The SpaCy language model to use (default:"en_core_web_sm")
        """
        self.nlp=spacy.load(language_model)

    def polished_answer(self,raw_response):
        """
        Polish a raw response into a user-friendly answer.
        Args:
        - raw_response(str): The raw response to polish
        Returns: 
        - str: The polished answere
        """
        doc = self.nlp(response)
        polished_answer = raw_response + "."

        return polished_answer

answer_tool = Tool(
    name = "PolishedResponses",
    func = Answering_Tool,
    description = "Modifies the raw responses obtained from order query tool into polished user-friendly responses.")
