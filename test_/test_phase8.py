"""
Phase 8 Verification: LLM Generation with LangChain
"""

from retrieval.simple_retriever import create_simple_retriever
from augmentation.prompt_templates import QA_PROMPT, QA_PROMPT_WITH_CITATIONS, format_docs_for_prompt
from generation.llm_client import create_llm, create_llm_chain
from generation.citation_handler import extract_citations, add_source_info

print("=" * 60)
print("Phase 8: LLM Generation with LangChain")
print("=" * 60)

video_id = "kY14KfZQ1TI"
question = "What is deep learning?"

# Step 1: Retrieve (Phase 6)
print("\nStep 1: Retrieving relevant chunks...")
retriever = create_simple_retriever(video_id=video_id, top_k=3)
docs = retriever.invoke(question)
print(f"  ✓ Retrieved {len(docs)} chunks")

# Step 2: Format context (Phase 7)
print("\nStep 2: Formatting context...")
context = format_docs_for_prompt(docs, include_chunk_ids=False)
print(f"  ✓ Formatted context: {len(context)} characters")

# Step 3: Create prompt (Phase 7)
print("\nStep 3: Creating prompt...")
prompt_text = QA_PROMPT.format(context=context, question=question)
print(f"  ✓ Created prompt: {len(prompt_text)} characters")

# Step 4: Generate answer without citations (Phase 8)
print("\nStep 4: Generating answer (no citations)...")
llm = create_llm(temperature=0.2, max_tokens=300)
response = llm.invoke(prompt_text)
answer = response.content

print(f"  ✓ Generated answer: {len(answer)} characters")
print(f"\n  Question: {question}")
print(f"\n  Answer:\n  {answer}")

# Step 5: Generate answer WITH citations
print("\n" + "=" * 60)
print("Step 5: Generating answer WITH citations...")
context_with_ids = format_docs_for_prompt(docs, include_chunk_ids=True)
prompt_citations = QA_PROMPT_WITH_CITATIONS.format(context=context_with_ids, question=question)

response_citations = llm.invoke(prompt_citations)
answer_citations = response_citations.content

print(f"  ✓ Generated answer with citations")
print(f"\n  Answer:\n  {answer_citations}")

# Step 6: Extract and validate citations
print("\n" + "=" * 60)
print("Step 6: Extracting citations...")
citations = extract_citations(answer_citations)
print(f"  ✓ Found {len(citations)} citations: {citations}")

# Add source info
result = add_source_info(answer_citations, docs)
print(f"  ✓ Mapped to sources: {result['num_valid_citations']} valid")

if result['sources']:
    print(f"\n  Source details:")
    for source in result['sources']:
        print(f"    - Chunk {source['chunk_id']} from video {source['video_id']}")
        print(f"      Text: {source['text'][:80]}...")

# Step 7: Test LLM chain with parser
print("\n" + "=" * 60)
print("Step 7: Testing LLM chain (with StrOutputParser)...")
llm_chain = create_llm_chain(temperature=0.2)
answer_parsed = llm_chain.invoke(prompt_text)

print(f"  ✓ LLM chain returned string directly")
print(f"  Type: {type(answer_parsed)}")
print(f"  Length: {len(answer_parsed)} characters")

print("\n" + "=" * 60)
print("Phase 8 Complete - LangChain Generation Working!")
print("=" * 60)
print("\nWhat We Can Do Now:")
print("  ✓ Generate answers using ChatOpenAI")
print("  ✓ Request and parse citations")
print("  ✓ Use StrOutputParser for clean string output")
print("  ✓ LLM chain ready for Phase 9 integration")
print("\nLangChain Components Used:")
print("  ✓ ChatOpenAI (LLM)")
print("  ✓ StrOutputParser (output parsing)")
print("\nReady for Phase 9: Complete QA chain!")