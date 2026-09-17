from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """Split extracted documents into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, text: str) -> list[str]:
        """Split text into chunks."""
        return self.splitter.split_text(text)