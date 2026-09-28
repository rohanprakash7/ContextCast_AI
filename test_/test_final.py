# test_final.py
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from openai import OpenAI
from config.settings import settings

print("Testing RAGAS 0.4.3 metric initialization...")

try:
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    ragas_llm = llm_factory('gpt-4o-mini', client=openai_client)
    ragas_embeddings = OpenAIEmbeddings(
        model='text-embedding-3-small',
        client=openai_client
    )
    
    f = Faithfulness(llm=ragas_llm)
    a = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    
    print("✓ SUCCESS! Metrics initialized correctly")
    print(f"  Faithfulness: {type(f)}")
    print(f"  AnswerRelevancy: {type(a)}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()