class Prompts:
    """
    A class that creates formatted prompts for generating answers based on given documents and questions.

    Methods:
        __init__(): Initializes the Prompts class.
        qa_ref(docs: list = [], q: str = None): Creates a formatted prompt for generating an answer based on the given documents and question.
    """

    def __init__(self):
        return

    def qa_ref(self, docs: list = [], q: str = None):
        """
        Create a formatted prompt for generating an answer based on the given documents and question.

        Args:
            docs (list): A list of dictionaries representing the documents. Each dictionary should have a "title" key and a "content" key.
            q (str): The question for which an answer is to be generated.

        Returns:
            str: A formatted prompt string.
        """
        # Join the content of the documents
        doc_content = '\n'.join([f"Document[name={doc['title']}] {doc['content']}" for doc in docs])

        # Replace placeholders in the template string with the joined documents and the question
        prompt = f"Create a concise and informative answer (no more than 200 words) for a given question based solely on the given documents. You must only use information from the given documents. Use an unbiased and journalistic tone. Do not repeat text. Cite the documents using Document[name] notation. If multiple documents contain the answer, cite those documents like ‘as stated in Document[name], Document[name], etc.’. If the documents do not contain the answer to the question, say that ‘answering is not possible given the available information.’\n{doc_content}\nQuestion: {q}; Answer:"

        return prompt

        Returns:
            str: A formatted string representing a prompt for generating an answer based on the given documents and question.
        """
        docs = '\n'.join([f"Document[name={d}]\ {t}\n\n" for (t, d, c) in [tuple(x.values()) for x in docs]])
        return """Create a concise and informative answer (no more than 200 words) for a given question based solely on the given documents. You must only use information from the given documents. Use an unbiased and journalistic tone. Do not repeat text. Cite the documents using Document[name] notation. If multiple documents contain the answer, cite those documents like ‘as stated in Document[name], Document[name], etc.’. If the documents do not contain the answer to the question, say that ‘answering is not possible given the available information.’
    [DOCUMENTS]
    Question: [QUERY]; Answer: """ \
                .replace('[DOCUMENTS]', docs) \
                .replace('[QUERY]', q)