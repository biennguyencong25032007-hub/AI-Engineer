from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextSplitter:
    def split(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        return splitter.split_documents(docs)
