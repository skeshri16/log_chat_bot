from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk(file):
    """Read uploaded txt file and chunk it."""
    text = file.read().decode("utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500
    )
    chunks = splitter.split_text(text)
    return chunks
