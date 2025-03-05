from langgraph.graph import END, StateGraph
from models import AgentCoder
from agents import create_agents

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
            'tests': {},
            'success': False
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
                'tests': state['tests'],
                'success': False
            }
            
        refine_code_ = refine_code.invoke({'code': code, 'error': errors})
        return {
            'code': refine_code_.code,
            'errors': None,
            'requirement': state['requirement'],
            'retry_count': retry_count,
            'tests': state['tests'],
            'success': False
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
            return {
                'code': executable_code.code,
                'errors': None,
                'requirement': state['requirement'],
                'tests': state['tests'],
                'retry_count': state.get('retry_count', 0),
                'success': True
            }
        except Exception as e:
            print('Found Error While Running')
            error = f"Execution Error : {e}"
            print(error)
            return {
                'code': executable_code.code,
                'errors': error,
                'requirement': state['requirement'],
                'tests': state['tests'],
                'retry_count': state.get('retry_count', 0),
                'success': False
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
            'errors': None,
            'success': False
        }

    def decide_to_end(state):
        print(f'Entering in Decide to End')
        retry_count = state.get('retry_count', 0)
        success = state.get('success', False)
        
        if success:
            print("All tests passed successfully. Exiting workflow.")
            return 'end'
        elif state.get('errors') and retry_count <= 3:
            return 'debugger'
        else:
            print("Maximum retries reached or no errors to fix. Exiting workflow.")
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

    # Add conditional edges
    workflow.add_conditional_edges(
        "executer",
        decide_to_end,
        {
            "end": END,
            "debugger": "debugger",
        },
    )

    # Add final edge to end
    workflow.add_edge("decide_to_end", END)

    return workflow.compile() 