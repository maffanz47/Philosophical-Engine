import fitz  # PyMuPDF
import re
from sqlalchemy.ext.asyncio import AsyncSession
from ..models.book import Book
from ..models.training_chunk import TrainingChunk

async def extract_chunks_from_pdf(book_id: str, filepath: str, philosopher: str, db: AsyncSession):
    doc = fitz.open(filepath)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    # Clean text: remove excessive whitespace, page numbers, headers
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r'[ \t]+', ' ', full_text)

    # Split into sentences first, then group into chunks of 150-300 words
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_word_count + word_count > 300 and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    # Save chunks to DB
    for i, chunk_text in enumerate(chunks):
        if len(chunk_text.strip()) < 50:  # skip tiny chunks
            continue
        chunk = TrainingChunk(
            book_id=book_id,
            text=chunk_text.strip(),
            label=philosopher,
            chunk_index=i
        )
        db.add(chunk)

    await db.commit()
    return len(chunks)