import pytest
import os
import json
from typing import Dict
from deepeval import assert_test
from deepeval.metrics import BiasMetric
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
bias = BiasMetric(threshold=0.5)
# Reason behind commenting out metric is expected_output cannot be None
contextual_precision = ContextualPrecisionMetric(threshold=0.5)
contextual_recall = ContextualRecallMetric(threshold=0.5)
answer_relevancy = AnswerRelevancyMetric(threshold=0.5)
faithfulness = FaithfulnessMetric(threshold=0.5)
evaluation_metrics = [
  bias,
  #contextual_precision,
  #contextual_recall,
  answer_relevancy,
  #faithfulness
]

from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import Settings

documents = SimpleDirectoryReader('documents').load_data()
Settings.chunk_size = 512
Settings.chunk_overlap = 50
vector_store = MilvusVectorStore(dim=1536, collection_name="quick_setup")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
llm = OpenAI(model="gpt-3.5-turbo")
rag_application = index.as_query_engine(llm=llm)

file_path = "deepEval_datasets/evaluation_data/test1.json"
with open(file_path, "r") as json_file:
    json_data = json.load(json_file)


@pytest.mark.parametrize(
    "input_output_pair",
    json_data,
)
def test_llamaindex(input_output_pair: Dict):
    input = input_output_pair.get("input", None)
    expected_output = input_output_pair.get("expected_output", None)
    response_object = rag_application.query(input)


    if response_object is not None:
        actual_output = response_object.response
        retrieval_context = [node.get_content() for node in response_object.source_nodes]

    actual_output = actual_output
    retrieval_context = retrieval_context
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected_output
    )
    assert_test(test_case, evaluation_metrics)
    deepeval_db = DeepEvalDatabase(hostname, database, username, password, port)
    deepeval_db.connection()
    directory_path = os.environ['DEEPEVAL_RESULTS_FOLDER']
    renamed_filenames = deepeval_db.rename_files_to_json(directory_path)
    for filename in renamed_filenames:
        json_file_path = os.path.join(directory_path, filename)
        deepeval_db.insert_data(json_file_path)

