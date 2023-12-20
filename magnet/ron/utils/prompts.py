class Prompts:
    def __init__(self):
        return
    def qa_ref(self, docs: list = [], q: str = None):
        docs = '\n'.join([f"Document: {d}\nText: {t}" for (t, d, c) in [tuple(x.values()) for x in docs]])
        return """Create a concise and informative answer (no more than 50 words) for a given question based solely on the given documents. You must only use information from the given documents. Use an unbiased and journalistic tone. Do not repeat text. Cite the documents using Document[name] notation. If multiple documents contain the answer, cite those documents like ‘as stated in Document[name], Document[name], etc.’. If the documents do not contain the answer to the question, say that ‘answering is not possible given the available information.’
[DOCUMENTS]
Question: [QUERY]; Answer: """ \
                .replace('[DOCUMENTS]', docs) \
                .replace('[QUERY]', q)

        