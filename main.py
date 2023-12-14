import logging
import asyncio as aio
import sys
from agentopy import Agent, Environment
from config import CONFIG

from llmagents.language.openai_language_models import OpenAIChatModel
from llmagents.language.embedding_models import OpenAIEmbeddingModel
from llmagents.language.llm_policy import LLMPolicy
from llmagents.db.in_memory_vector_db import InMemoryVectorDB
from llmagents.components import TodoList, Creativity, Email, Memory, WebBrowser

logging.basicConfig(handlers=[logging.FileHandler('llmagents.log', mode='w', encoding='utf-8'), logging.StreamHandler(sys.stdout)],
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logging.info("Running llmagents")

logger = logging.getLogger('main')


async def main():
    """
    Main function of the project. It is responsible for running the assitant.
    """
    language_model, embeddings_model = None, None

    if CONFIG.agent.language_model.startswith("openai:"):
        language_model = OpenAIChatModel(CONFIG.openai_api_key,
                                         CONFIG.agent.language_model.lstrip("openai:"), json=True)

    if CONFIG.agent.embeddings_model.startswith("openai:"):
        embeddings_model = OpenAIEmbeddingModel(CONFIG.openai_api_key,
                                                CONFIG.agent.embeddings_model.lstrip("openai:"))

    if not language_model or not embeddings_model:
        raise Exception("No language model or embeddings model was provided")

    todo_list = TodoList()
    vector_db = InMemoryVectorDB(embeddings_model.dim(), 'cosine')

    env = Environment(
        [
            ("Todo List", todo_list),
            ("Creativity", Creativity(language_model=OpenAIChatModel(CONFIG.openai_api_key,
                                                                     CONFIG.agent.language_model.lstrip("openai:"), json=False))),
            ("Email", Email(
                imap_address=CONFIG.email_imap_address,
                smtp_address=CONFIG.email_smtp_address,
                smtp_port=CONFIG.email_smtp_port,
                login=CONFIG.email_login,
                password=CONFIG.email_password,
                from_address=CONFIG.email_from,
                outbound_emails_whitelist=CONFIG.outbound_emails_whitelist,
            )),
            ("Web Browser", WebBrowser(language_model=OpenAIChatModel(CONFIG.openai_api_key,
                                                                      CONFIG.agent.language_model.lstrip("openai:"), json=False)))
        ]
    )

    policy = LLMPolicy(
        language_model,
        CONFIG.agent.system_prompt.format(
            name="Dude", **{f'my_{k}': v for k, v in CONFIG.me.__dict__.items()}),
        CONFIG.agent.response_template
    )
    memory = Memory(vector_db, embeddings_model, 10)

    agent = Agent(
        policy,
        env,
        [
            memory
        ]
    )

    tasks = []

    while True:

        if not tasks:
            inp = input("User: ")

            if inp == "start":
                tasks.append(env.start())
                tasks.append(agent.start())
            elif inp.startswith("mem"):
                await memory.add(inp.lstrip("mem "))
            elif inp.startswith("todo"):
                await todo_list.add(inp.lstrip("todo "))
        else:
            e = await aio.wait(tasks, return_when=aio.FIRST_EXCEPTION)
            import traceback
            print(''.join(traceback.format_tb(
                e[0].pop().exception().__traceback__)))
            tasks.clear()

if __name__ == "__main__":
    aio.run(main())
