from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnablePassthrough

from app.search.web import chain as search_chain
from app.writer import chain as writer_chain

chain_notypes = (
    RunnablePassthrough().assign(research_summary=search_chain) | writer_chain
)


class InputType(BaseModel):
    question: str


chain = chain_notypes.with_types(input_type=InputType)
