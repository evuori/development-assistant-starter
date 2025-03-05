from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from models import Code, Test, ExecutableCode, RefineCode

def setup_environment():
    """Setup environment variables and LLM"""
    from dotenv import load_dotenv
    import os
    
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

*REQURIEMENT*
{requirement}'''
    )

    # Test generation prompt
    test_gen_prompt = ChatPromptTemplate.from_template(
        '''**Role**: As a tester, your task is to create Basic and Simple test cases based on provided Requirement and Python Code. 
These test cases should encompass Basic, Edge scenarios to ensure the code's robustness, reliability, and scalability.

**CRITICAL FORMAT REQUIREMENTS**:
1. Input MUST be a list of lists: [[test1_inputs], [test2_inputs], ...]
2. Output MUST be a list of lists: [[test1_output], [test2_output], ...]
3. Each output MUST be a list containing exactly one value
4. The number of input test cases MUST match the number of output test cases

Examples:

1. For a function that adds two numbers:
Input: [[1, 2], [3, 4]]  # Two test cases
Output: [[3], [7]]       # Each output is a list with one value

2. For a function that calculates average:
Input: [[1, 2, 3], [4, 5, 6]]  # Two test cases
Output: [[2.0], [5.0]]         # Each output is a list with one value

3. For a function that returns a string:
Input: [["hello"], ["world"]]  # Two test cases
Output: [["HELLO"], ["WORLD"]] # Each output is a list with one value

**1. Basic Test Cases**:
- **Objective**: Basic and Small scale test cases to validate basic functioning 
**2. Edge Test Cases**:
- **Objective**: To evaluate the function's behavior under extreme or unusual conditions.

**Instructions**:
- Implement a comprehensive set of test cases based on requirements
- Pay special attention to edge cases as they often reveal hidden bugs
- Only Generate Basics and Edge cases which are small
- Avoid generating Large scale and Medium scale test case
- Focus only small, basic test-cases
- CRITICAL: Each output MUST be a list containing exactly one value
- CRITICAL: The number of input and output test cases MUST match
- CRITICAL: All values must be properly formatted as lists

*REQURIEMENT*
{requirement}
**Code**
{code}
'''
    )

    # Execution prompt
    python_execution_gen = ChatPromptTemplate.from_template(
        """You have to add testing layer in the *Python Code* that can help to execute the code. You need to pass only Provided Input as argument and validate if the Given Expected Output is matched.

*Instructions*:
- Make sure to return the error if the assertion fails
- Generate the code that can be execute
- For floating-point comparisons, use math.isclose() with appropriate relative and absolute tolerances
- Example: math.isclose(result, expected, rel_tol=1e-9, abs_tol=0.0)

Python Code to execute:
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
    
    *Code*: {code}
    *Error*: {error}
    """
    )

    # Create the agents
    coder = create_structured_output_runnable(Code, llm, code_gen_prompt)
    tester_agent = create_structured_output_runnable(Test, llm, test_gen_prompt)
    execution = create_structured_output_runnable(ExecutableCode, llm, python_execution_gen)
    refine_code = create_structured_output_runnable(RefineCode, llm, python_refine_gen)

    return coder, tester_agent, execution, refine_code 