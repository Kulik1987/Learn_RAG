from langchain.schema import Document
from typing import Iterator

class TextLoaderWithEncoding:
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                text = f.read()
            yield Document(page_content=text)
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

    def load(self) -> list[Document]:
        return list(self.lazy_load())
