from pydantic import BaseModel, Field, ConfigDict, validator
from typing import List, Any, Dict, Optional, TypedDict, Union

class Code(BaseModel):
    """Plan to follow in future"""
    code: str = Field(description="Detailed optimized error-free Python code on the provided requirements")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_serialization_defaults=True
    )

class Test(BaseModel):
    """Test cases for code validation"""
    Input: List[List[Any]] = Field(description="Input for Test cases to evaluate the provided code")
    Output: List[List[Any]] = Field(description="Expected Output for Test cases to evaluate the provided code")
    
    @validator('Output')
    def validate_output(cls, v):
        """Validate that each output is a list containing a single value"""
        if not all(isinstance(x, list) for x in v):
            raise ValueError("Each output must be a list")
        if not all(len(x) == 1 for x in v):
            raise ValueError("Each output list must contain exactly one value")
        return v
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_serialization_defaults=True
    )

class ExecutableCode(BaseModel):
    """Plan to follow in future"""
    code: str = Field(description="Detailed optimized error-free Python code with test cases assertion")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_serialization_defaults=True
    )

class RefineCode(BaseModel):
    code: str = Field(description="Optimized and Refined Python code to resolve the error")
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_serialization_defaults=True
    )

class AgentCoder(TypedDict):
    requirement: str
    code: str
    tests: Dict[str, Any]
    errors: Optional[str]
    retry_count: int 