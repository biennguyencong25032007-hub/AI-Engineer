from langchain.document_loaders import TextLoader

class DocumentLoader:
    def load(self, path):
        loader = TextLoader(path)
        return loader.load()