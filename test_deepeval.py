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
  contextual_precision,
  contextual_recall,
  answer_relevancy,
  faithfulness
]

from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

from deepeval_database import DeepEvalDatabase

hostname = os.environ['HOSTNAME']
database = os.environ['DATABASE']
username = os.environ['USERNAME']
password = os.environ['PASSWORD']
port = os.environ['PORT']


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

from llama_index.core import PromptTemplate

text_qa_template_str = (
    "Context information is"
    " below.\n---------------------\n{context_str}\n---------------------\nUsing"
    " both the context information and also using your own knowledge, answer"
    " the question: {query_str}\nIf the context isn't helpful, you can also"
    " answer the question on your own.\n"
)
text_qa_template = PromptTemplate(text_qa_template_str)

refine_template_str = (
    "The original question is as follows: {query_str}\nWe have provided an"
    " existing answer: {existing_answer}\nWe have the opportunity to refine"
    " the existing answer (only if needed) with some more context"
    " below.\n------------\n{context_msg}\n------------\nUsing both the new"
    " context and your own knowledge, update or repeat the existing answer.\n"
)
refine_template = PromptTemplate(refine_template_str)

prompt_output = index.as_query_engine(
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        llm=llm,
    )
    
file_path = "deepEval_datasets/evaluation_data/20240424_175658.json"
with open(file_path, "r") as json_file:
    json_data = json.load(json_file)
    
@pytest.mark.parametrize(
    "input_output_pair", 
    json_data,
)
def test_llamaindex(input_output_pair: Dict):
    input = input_output_pair.get("input", None)
    response_object = rag_application.query(input)
    
    if response_object is not None:
        actual_output = response_object.response
        retrieval_context = [node.get_content() for node in response_object.source_nodes]  
    actual_output = actual_output
    retrieval_context = retrieval_context
    
    expected =prompt_output.query(input)
    if expected is not None:
        expected_output = expected.response
    expected_output=expected_output
    
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected_output
    )
    assert_test(test_case, evaluation_metrics)


import deepeval
@deepeval.on_test_run_end
def function_to_be_called_after_test_run():
    directory_path = os.environ['DEEPEVAL_RESULTS_FOLDER']
    deepeval_db = DeepEvalDatabase(hostname, database, username, password, port)
    deepeval_db.connection()
    json_file_path = deepeval_db.rename_file_to_json(directory_path)
    deepeval_db.insert_data(json_file_path)
    print("Test finished Pavan")



