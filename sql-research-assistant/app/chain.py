from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnablePassthrough

from app.search.web import chain as search_chain
from app.writer import chain as writer_chain

from dotenv import load_dotenv
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

load_dotenv()
wait_for_all_tracers()

chain_notypes = (
        RunnablePassthrough().assign(research_summary = search_chain) | writer_chain
)


class InputType(BaseModel):
    question: str


chain = chain_notypes.with_types(input_type = InputType)

if __name__ == "__main__":
    input_data = {
        "question": "Who is older? Point guards or Centers?"
    }

    print(chain.invoke(input_data))
