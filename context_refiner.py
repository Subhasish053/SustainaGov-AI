# REFINEMENT RULES FOR CONTEXT EXTRACTION


def refine_context(retrieved_docs, max_chars=1200):
    """
    Refines retrieved FAISS documents into a clean, focused context.
    - Prioritizes the most relevant (first) document
    - Groups related information
    - Limits total context size
    """

    if not retrieved_docs:
        return ""

    # 1️⃣ Primary scheme = first retrieved document
    primary_doc = retrieved_docs[0].page_content.strip()

    refined_context = primary_doc
    current_length = len(primary_doc)

    # 2️⃣ Add secondary context (if space allows)
    for doc in retrieved_docs[1:]:
        content = doc.page_content.strip()

        # Avoid duplication
        if content in refined_context:
            continue

        if current_length + len(content) <= max_chars:
            refined_context += "\n\nRelated Information:\n" + content
            current_length += len(content)
        else:
            break

    return refined_context