import os
from dotenv import load_dotenv
import streamlit as st
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field, ConfigDict
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, Any, Dict, Optional, Sequence, TypedDict, List, Tuple
from langgraph.graph import END, StateGraph
from agents import setup_environment, create_agents
from workflow import create_workflow

# Define the Code model
class Code(BaseModel):
    """Plan to follow in future"""
    code: str = Field(description="Detailed optimized error-free Python code on the provided requirements")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_serialization_defaults=True
    )

# Define the Test model
class Test(BaseModel):
    """Plan to follow in future"""
    Input: List[List[Any]] = Field(description="Input for Test cases to evaluate the provided code")
    Output: List[Any] = Field(description="Expected Output for Test cases to evaluate the provided code")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_serialization_defaults=True
    )

# Define the ExecutableCode model
class ExecutableCode(BaseModel):
    """Plan to follow in future"""
    code: str = Field(description="Detailed optimized error-free Python code with test cases assertion")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_serialization_defaults=True
    )

# Define the RefineCode model
class RefineCode(BaseModel):
    code: str = Field(description="Optimized and Refined Python code to resolve the error")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_serialization_defaults=True
    )

# Define the AgentCoder state
class AgentCoder(TypedDict):
    requirement: str
    code: str
    tests: Dict[str, Any]
    errors: Optional[str]
    retry_count: int

def setup_environment():
    """Setup environment variables and LLM"""
    # Load environment variables from .env file
    load_dotenv()
    
    # Environment variables are now loaded from .env file
    return AzureChatOpenAI(
        temperature=0,
        max_tokens=1024,
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4-1106-preview")
    )

def create_agents(llm):
    """Create all the necessary agents"""
    # Code generation prompt
    code_gen_prompt = ChatPromptTemplate.from_template(
        '''**Role**: You are a expert software python programmer. You need to develop python code
**Task**: As a programmer, you are required to complete the function. Use a Chain-of-Thought approach to break
down the problem, create pseudocode, and then write the code in Python language. Ensure that your code is
efficient, readable, and well-commented.

**Instructions**:
1. **Understand and Clarify**: Make sure you understand the task.
2. **Algorithm/Method Selection**: Decide on the most efficient way.
3. **Pseudocode Creation**: Write down the steps you will follow in pseudocode.
4. **Code Generation**: Translate your pseudocode into executable Python code
5. **Important**: Always define the function first, then write the test cases
6. **Important**: Make sure the function name matches exactly what is being called in the test cases

*REQURIEMENT*
{requirement}'''
    )

    # Test generation prompt
    test_gen_prompt = ChatPromptTemplate.from_template(
        '''**Role**: As a tester, your task is to create Basic and Simple test cases based on provided Requirement and Python Code. 
These test cases should encompass Basic, Edge scenarios to ensure the code's robustness, reliability, and scalability.
**1. Basic Test Cases**:
- **Objective**: Basic and Small scale test cases to validate basic functioning 
**2. Edge Test Cases**:
- **Objective**: To evaluate the function's behavior under extreme or unusual conditions.
**Instructions**:
- Implement a comprehensive set of test cases based on requirements.
- Pay special attention to edge cases as they often reveal hidden bugs.
- Only Generate Basics and Edge cases which are small
- Avoid generating Large scale and Medium scale test case. Focus only small, basic test-cases
- Make sure to use the exact function name that was defined in the code

*REQURIEMENT*
{requirement}
**Code**
{code}
'''
    )

    # Execution prompt
    python_execution_gen = ChatPromptTemplate.from_template(
        """You have to add testing layer in the *Python Code* that can help to execute the code. You need to pass only Provided Input as argument and validate if the Given Expected Output is matched.
*Instruction*:
- Make sure to return the error if the assertion fails
- Generate the code that can be execute
- Ensure the function definition is included before the test cases
- Make sure the function name matches exactly what is being called

Python Code to excecute:
*Python Code*:{code}
Input and Output For Code:
*Input*:{input}
*Expected Output*:{output}"""
    )

    # Refinement prompt
    python_refine_gen = ChatPromptTemplate.from_template(
        """You are expert in Python Debugging. You have to analysis Given Code and Error and generate code that handles the error
    *Instructions*:
    - Make sure to generate error free code
    - Generated code is able to handle the error
    - Ensure the function definition is included before the test cases
    - Make sure the function name matches exactly what is being called
    
    *Code*: {code}
    *Error*: {error}
    """
    )

    # Create the agents using create_structured_output_runnable for compatibility with current API version
    coder = create_structured_output_runnable(Code, llm, code_gen_prompt)
    tester_agent = create_structured_output_runnable(Test, llm, test_gen_prompt)
    execution = create_structured_output_runnable(ExecutableCode, llm, python_execution_gen)
    refine_code = create_structured_output_runnable(RefineCode, llm, python_refine_gen)

    return coder, tester_agent, execution, refine_code

def create_workflow(coder, tester_agent, execution, refine_code):
    """Create the workflow graph"""
    def programmer(state):
        print(f'Entering in Programmer')
        requirement = state['requirement']
        code_ = coder.invoke({'requirement': requirement})
        return {
            'code': code_.code, 
            'requirement': requirement,
            'retry_count': 0,
            'errors': None,
            'tests': {}
        }

    def debugger(state):
        print(f'Entering in Debugger')
        errors = state['errors']
        code = state['code']
        retry_count = state.get('retry_count', 0) + 1
        
        if retry_count > 3:  # Maximum retry limit
            print("Maximum retry limit reached. Stopping debug cycle.")
            return {
                'code': code,
                'errors': None,
                'requirement': state['requirement'],
                'retry_count': retry_count,
                'tests': state['tests']
            }
            
        refine_code_ = refine_code.invoke({'code': code, 'error': errors})
        return {
            'code': refine_code_.code,
            'errors': None,
            'requirement': state['requirement'],
            'retry_count': retry_count,
            'tests': state['tests']
        }

    def executer(state):
        print(f'Entering in Executer')
        tests = state['tests']
        input_ = tests['input']
        output_ = tests['output']
        code = state['code']
        executable_code = execution.invoke({"code": code, "input": input_, 'output': output_})
        error = None
        try:
            exec(executable_code.code)
            print("Code Execution Successful")
        except Exception as e:
            print('Found Error While Running')
            error = f"Execution Error : {e}"
            print(error)
        return {
            'code': executable_code.code,
            'errors': error,
            'requirement': state['requirement'],
            'tests': state['tests'],
            'retry_count': state.get('retry_count', 0)
        }

    def tester(state):
        print(f'Entering in Tester')
        requirement = state['requirement']
        code = state['code']
        tests = tester_agent.invoke({'requirement': requirement, 'code': code})
        return {
            'tests': {'input': tests.Input, 'output': tests.Output},
            'requirement': requirement,
            'code': state['code'],
            'retry_count': state.get('retry_count', 0),
            'errors': None
        }

    def decide_to_end(state):
        print(f'Entering in Decide to End')
        retry_count = state.get('retry_count', 0)
        if state.get('errors') and retry_count <= 3:
            return 'debugger'
        return 'end'

    # Create the workflow
    workflow = StateGraph(AgentCoder)

    # Add nodes
    workflow.add_node("programmer", programmer)
    workflow.add_node("debugger", debugger)
    workflow.add_node("executer", executer)
    workflow.add_node("tester", tester)

    # Build graph
    workflow.set_entry_point("programmer")
    workflow.add_edge("programmer", "tester")
    workflow.add_edge("debugger", "executer")
    workflow.add_edge("tester", "executer")

    workflow.add_conditional_edges(
        "executer",
        decide_to_end,
        {
            "end": END,
            "debugger": "debugger",
        },
    )

    return workflow.compile()

def main():
    st.title("MAD Machine Assisted Development")
    st.subheader("by Erno Vuori (erno.vuori@almamedia.fi)")
    
    # Setup environment and create agents
    llm = setup_environment()
    coder, tester_agent, execution, refine_code = create_agents(llm)
    app = create_workflow(coder, tester_agent, execution, refine_code)

    # Get user input
    requirement = st.text_area("Enter your coding requirement:", height=150)
    
    if st.button("Generate Code"):
        if requirement:
            with st.spinner("Generating code..."):
                # Run the workflow
                config = {"recursion_limit": 50}
                inputs = {"requirement": requirement}
                running_dict = {}
                
                for event in app.stream(inputs, config=config):
                    for k, v in event.items():
                        running_dict[k] = v
                        if k != "__end__":
                            st.write(v)
                            st.write('----------' * 20)
                
                # Display final results
                if 'code' in running_dict:
                    st.subheader("Generated Code:")
                    st.code(running_dict['code'], language='python')
        else:
            st.warning("Please enter a requirement first.")

if __name__ == "__main__":
    main() 