from dotenv import load_dotenv
from langchain.agents import create_openai_tools_agent
from langchain.agents.agent import AgentExecutor
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.prompt import SQL_FUNCTIONS_SUFFIX
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()
message_history = ChatMessageHistory()

# import sqlite3
# from langchain.chains import create_sql_query_chain
# from langchain.agents import AgentExecutor, create_openai_functions_agent
# from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
# from langchain_community.utilities import SQLDatabase

prompt_system_message = """
You are a helpful assistant who both can reply to customers with helpful text, and also can make sql queries
on the specified schema.
You can make a conversation with the user and understand what he needs and convert that to a database query.
You mainly interact with the specified schema and select the correct table to interact with.
You also help users make TODO lists, as you save the tasks on the database on the date the user specifies.
You can retrieve the relevant tasks according to the user input.
Always to reply to the user with helpful text and consider he don't understand the scheam or SQL, just reply in text.
Never mention any information about the database schema, tables or fields just interact with it silently.
Format your output as descriptive text and never write any queries or database information.
Never format the output as tables, just write the titles and dates.
Never retrieve all the data from the table, just retrieve the needed entries.
Never retrieve all the columns from tables, just select the needed ones.
In case the user didn't ask you anything related to the database, insertion or retrieval, never talk about it, just
reply to him with chat completion.
"""


def create_tasks_table(connection):
    cursor = connection.cursor()
    # cursor.execute("""DROP TABLE IF EXISTS tasks;""")
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task TEXT NOT NULL,
                        due_date TIMESTAMP NOT NULL,
                        completed BOOLEAN DEFAULT FALSE,
                        created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )"""
    )
    connection.commit()


# Function to insert a task into the tasks table
def insert_task(connection, task, due_date):
    cursor = connection.cursor()
    cursor.execute("INSERT INTO tasks (task, due_date) VALUES (?, ?)", (task, due_date))
    connection.commit()


# Function to retrieve all tasks from the tasks table
def get_all_tasks(connection):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM tasks")
    tasks = cursor.fetchall()
    return tasks


# Function to retrieve tasks based on due_date
def get_tasks_by_due_date(connection, due_date):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM tasks WHERE due_date = ?", (due_date,))
    tasks = cursor.fetchall()
    return tasks


def execute_query(connection, query: str):
    cursor = connection.cursor()
    cursor.execute(query)
    connection.commit()

    if "INSERT" in query:
        return cursor.fetchall()


def chatbot_interaction(text: str):
    # conn = sqlite3.connect("tasks.db")
    # create_tasks_table(conn)

    db = SQLDatabase.from_uri("sqlite:///tasks.db", sample_rows_in_table_info=3)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    # db_chain = create_sql_query_chain(llm, db)
    # query = db_chain.invoke({"question": text})

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    context = toolkit.get_context()
    tools = toolkit.get_tools()
    messages = [
        SystemMessage(content=prompt_system_message),
        AIMessage(content=SQL_FUNCTIONS_SUFFIX),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    prompt = prompt.partial(**context)
    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=toolkit.get_tools(),
        verbose=True,
    )
    # return agent_executor.invoke({"input": text})

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history.invoke({"input": text}, {"configurable": {"session_id": "[dummy-id]"}})

    # Old code using sql database chain that returns queries and using python functions
    # prompt = db_chain.get_prompts()[0].partial(table_info=db.get_context()["table_info"])
    # print(db_chain.get_prompts()[0].pretty_print())
    # q_chain = prompt | llm
    # print(q_chain.invoke({"input": "Add a task tomorrow for azd"}))

    # tools = [QuerySQLDataBaseTool]
    # agent = create_openai_functions_agent(llm, tools, prompt + "{agent_scratchpad}")
    # print(db.dialect)
    # print(db.get_usable_table_names())
    # print(db.run("SELECT * FROM tasks LIMIT 10;"))
    # Insert tasks into the tasks
    # insert_task(conn, "Buy groceries", "2024-02-14 10:00:00")
    # insert_task(conn, "Complete homework", "2024-02-15 18:00:00")
    # insert_task(conn, "Go for a run", "2024-02-16 08:00:00")
    # Retrieve tasks based on due date
    # tasks_due_on_15th = get_tasks_by_due_date(conn, "2024-02-15 18:00:00")
    # print("\nTasks due on 2024-02-15:")
    # for task in tasks_due_on_15th:
    #     print(task)

    # output = execute_query(conn, query)
    # print(query)
    # print(output)
    # Retrieve all tasks from the tasks
    # tasks = get_all_tasks(conn)
    # print("All tasks in the tasks:")
    # for task in tasks:
    #     print(task)

    # Close the database connection
    # conn.close()


def main():
    print("Hello, How can I help you!")
    while True:
        user_text = input("You: ")
        if user_text.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        else:
            response = chatbot_interaction(user_text)
            print("Assistant:", response.get("reply", response.get("output")))


if __name__ == "__main__":
    main()
