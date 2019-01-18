import sys, time
from lp_utils import Data, convert_PSL_to_LP, saucy
from solve_lp import solve_lp_gurobi

train_file = sys.argv[1]#'./trust_pred.txt'
train_data = Data(train_file)
train_data.read()

A, b, c = convert_PSL_to_LP(train_data)
b = b.toarray()
c = c.toarray()
start = time.time()
objective, solution = solve_lp_gurobi(A, b, c)
end = time.time()
print('Time taken to solve : {}sec'.format(end-start))
print(objective)
'''
LA, Lb, Lc, LG, Lh, compresstime, Bcc = saucy.liftAbc(A, b, c, G=A, h=b,
                                                      sparse=True, orbits=False)
start = time.time()
objective, solution = solve_lp_gurobi(LA, Lb, Lc)
end = time.time()
print('Time taken to solve lifted problem: {}sec'.format(end-start))
print(objective)
'''

