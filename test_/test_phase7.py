"""
Phase 7 Verification: LangChain Prompt Templates
"""

from retrieval.simple_retriever import create_simple_retriever
from augmentation.prompt_templates import (
    QA_PROMPT,
    QA_PROMPT_WITH_CITATIONS,
    format_docs_for_prompt,
    create_qa_chain_prompt
)

print("=" * 60)
print("Phase 7: LangChain Prompt Engineering")
print("=" * 60)

# Step 1: Retrieve relevant chunks
print("\nStep 1: Retrieving relevant chunks...")
video_id = "kY14KfZQ1TI"
question = "What is deep learning?"

retriever = create_simple_retriever(video_id=video_id, top_k=4)
docs = retriever.invoke(question)

print(f"  ✓ Retrieved {len(docs)} chunks for: '{question}'")

# Step 2: Format context without citations
print("\nStep 2: Formatting context (no citations)...")
context = format_docs_for_prompt(docs, include_chunk_ids=False)
print(f"  ✓ Formatted context: {len(context)} characters")
print(f"\n  Context preview:")
print(f"  {context[:200]}...")

# Step 3: Create prompt without citations
print("\nStep 3: Creating prompt without citations...")
prompt = QA_PROMPT.format(context=context, question=question)
print(f"  ✓ Created prompt: {len(prompt)} characters")
print(f"\n  Prompt preview:")
print(f"  {prompt[:300]}...")

# Step 4: Format context with citations
print("\nStep 4: Formatting context with citation IDs...")
context_with_ids = format_docs_for_prompt(docs, include_chunk_ids=True)
print(f"  ✓ Formatted with chunk IDs")
print(f"\n  Context with IDs preview:")
print(f"  {context_with_ids[:250]}...")

# Step 5: Create prompt with citations
print("\nStep 5: Creating prompt requesting citations...")
prompt_with_citations = QA_PROMPT_WITH_CITATIONS.format(
    context=context_with_ids,
    question=question
)
print(f"  ✓ Created citations prompt: {len(prompt_with_citations)} characters")

# Step 6: Test prompt template selector
print("\nStep 6: Testing prompt template selector...")
standard_prompt = create_qa_chain_prompt(include_citations=False)
citation_prompt = create_qa_chain_prompt(include_citations=True)

print(f"  ✓ Standard prompt selected: {standard_prompt == QA_PROMPT}")
print(f"  ✓ Citation prompt selected: {citation_prompt == QA_PROMPT_WITH_CITATIONS}")

# Summary
print("\n" + "=" * 60)
print("Phase 7 Complete - LangChain Prompts Ready!")
print("=" * 60)
print("\nWhat We Can Do Now:")
print("  ✓ Format retrieved chunks into LLM context")
print("  ✓ Create prompts with/without citations")
print("  ✓ Use LangChain PromptTemplate for consistency")
print("  ✓ Ready to combine with LLM in Phase 8")
print("\nLangChain Components Used:")
print("  ✓ PromptTemplate (for string templates)")
print("  ✓ Document schema (from retrieval)")
print("\nReady for Phase 8: LLM generation!")