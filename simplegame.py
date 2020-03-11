from pyquil import Program
import pyquil.api as api
from pyquil.gates import *

qvm = api.QVMConnection()
picard_register = 1
answer_register = 0

then_branch = Program(X(0))
else_branch = Program(I(0))

prog = (Program()
    .inst(X(0), H(1))
    .inst(H(0))
    .measure(1, picard_register)
    .inst(H(0))
    .measure(0, answer_register))


qvm.run(prog, [0, 1], trials=10)
