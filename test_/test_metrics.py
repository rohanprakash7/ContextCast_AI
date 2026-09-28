# test_metrics_check.py
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from openai import OpenAI
from config.settings import settings

openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
ragas_llm = llm_factory('gpt-4o-mini', client=openai_client)
ragas_embeddings = OpenAIEmbeddings(model='text-embedding-3-small', client=openai_client)

f = Faithfulness(llm=ragas_llm)

print("Metric object:", f)
print("Type:", type(f))
print("Has __class__:", hasattr(f, '__class__'))
print("Is BaseMetric?:", hasattr(f, 'adapt'))
print("Dir:", [x for x in dir(f) if not x.startswith('_')])