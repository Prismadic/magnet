class Prompts:
    """
    The `Prompts` class is used to generate a formatted prompt for generating an answer based on a given question and a list of documents.
    """

    def __init__(self):
        """
        Initializes an instance of the `Prompts` class.
        """
        return

    def qa_ref(self, docs: list = [], q: str = None) -> str:
        """
        Generates a formatted prompt for generating an answer based on the given documents and question.

        Args:
            docs (list): A list of dictionaries representing the documents. Each dictionary should have a "title" key and a "content" key.
            q (str): The question for which an answer is to be generated.

        Returns:
            str: A formatted string representing a prompt for generating an answer based on the given documents and question.
        """
        docs = '\n'.join([f"Document[name={d}]\ {t}\n\n" for (t, d, c) in [tuple(x.values()) for x in docs]])
        return """Create a concise and informative answer (no more than 200 words) for a given question based solely on the given documents. You must only use information from the given documents. Use an unbiased and journalistic tone. Do not repeat text. Cite the documents using Document[name] notation. If multiple documents contain the answer, cite those documents like ‘as stated in Document[name], Document[name], etc.’. If the documents do not contain the answer to the question, say that ‘answering is not possible given the available information.’
    [DOCUMENTS]
    Question: [QUERY]; Answer: """ \
                .replace('[DOCUMENTS]', docs) \
                .replace('[QUERY]', q)
    
    def classy(self, doc: str = "", q: str = None, schema={"original_data": "object", "scores": "list:float", "classes":"list:string"}) -> str:
        """
        Generates a formatted prompt for generating a classification based on the given documents and question.

        Args:
            doc (str): A text sample.
            q (str): The context for which a classification is to be generated.

        Returns:
            str: A formatted string representing a prompt for generating a classification based on the given documents and question.
        """
        prompt = f"{q}\nPlease ONLY reply in JSON (do not provide any output but JSON) and keep your schema consistent with the following: {schema}\n\nand provide a score for each class:\n\n{doc} \n\n and make sure to put the original data into the 'original data' key."
        return prompt