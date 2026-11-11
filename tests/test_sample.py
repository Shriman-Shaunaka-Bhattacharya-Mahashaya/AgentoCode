"""Public calculator behavior exercised by the example application."""
import importlib.util
from pathlib import Path
import pytest

@pytest.fixture
def calculator():
    path=Path(__file__).resolve().parents[1]/'sample_project/calculator.py'
    spec=importlib.util.spec_from_file_location('sample_calculator_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.SimpleCalculator()

def test_arithmetic_results_and_recorded_order(calculator):
    assert calculator.add(2,3)==5
    assert calculator.subtract(2,3)==-1
    assert calculator.multiply(-2,3)==-6
    assert calculator.divide(7,2)==3.5
    assert calculator.power(2,8)==256
    assert calculator.factorial(0)==1
    assert calculator.factorial(5)==120
    history=calculator.get_history()
    assert len(history)==7 and history[0]=='2 + 3 = 5' and history[-1]=='5! = 120'

def test_invalid_calculation_does_not_add_history(calculator):
    with pytest.raises(ValueError,match='zero'):calculator.divide(1,0)
    with pytest.raises(ValueError,match='negative'):calculator.factorial(-1)
    assert calculator.get_history()==[]

def test_history_is_the_same_mutable_list(calculator):
    result=calculator.get_history();result.append('external marker')
    assert calculator.get_history() is result and calculator.history==['external marker']
