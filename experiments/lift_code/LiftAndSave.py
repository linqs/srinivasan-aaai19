import sys, os
sys.path.append(os.path.abspath('..'))
import scipy.sparse as sps
import numpy as np
import reloop.saucy as saucy
import csv
import time
import decimal
decimal.getcontext().prec = 6


class Data:
    file_name = ''
    metadata_read = False
    weights = None
    data = None
    rows = None
    cols = None
    csr_data = None
    nnz = 0
    n_rows = 0
    n_cols = 0
    l_fn_pow = None
    pick_matrix = None
    col_nnz = None
    coeff = None
    def __init__(self, file_name=None, weights=None, csr_data=None, coeff=None, l_fn_pow=None):
        if (file_name):
            self.file_name = file_name
        elif (not(weights is None) and not(csr_data is None)
              and not(coeff is None) and not(l_fn_pow is None)):
            self.coeff = coeff
            self.weights = weights
            self.csr_data = csr_data
            self.n_cols = self.csr_data.shape[1]
            self.n_rows = self.csr_data.shape[0]
            self.pick_matrix = csr_data.copy()
            self.pick_matrix[self.pick_matrix.nonzero()] = 1
            self.col_nnz = np.array(self.pick_matrix.sum(axis=0)).reshape(-1)
            self.l_fn_pow = l_fn_pow
        else:
            print("Insufficient parameters passed. Pass either file name or all matricies.")

    def read_metadata(self, file_name):
        with open(self.file_name) as fp:
            for line in fp:
                line.strip()
                x, fn_pow = line.split('#')
                self.n_rows += 1
                x_comp = x.strip().split(' ')[1:]
                for c in x_comp:
                    col_ind = int(c.split(':')[0])
                    self.n_cols = col_ind if col_ind > self.n_cols else self.n_cols
                    self.nnz += 1
        self.metadata_read = True
        self.n_cols += 1
    def init(self):
        self.weights = np.zeros(self.n_rows)
        self.l_fn_pow = np.ones(self.n_rows)
        self.data = np.zeros(self.nnz)
        self.rows = np.zeros(self.n_rows+1)
        self.cols = np.zeros(self.nnz)
    def read(self):
        if not self.metadata_read:
            self.read_metadata(self.file_name)
        self.init()
        with open(self.file_name) as fp:
            row_ind = 0
            nnz = 0
            for line in fp:
                line.strip()
                x, fn_pow = line.split('#')
                x = x.strip()
                x_comp = x.split(' ')[1:]
                self.weights[row_ind] = float(x.split(' ')[0])
                self.l_fn_pow[row_ind] = float(fn_pow.strip().split('^')[1])
                for c in x_comp:
                    col_ind = int(c.split(':')[0])
                    col_val = float(c.split(':')[1])
                    self.data[nnz] = col_val
                    #self.rows[nnz] = row_ind
                    self.cols[nnz] = col_ind
                    nnz += 1
                row_ind += 1
                self.rows[row_ind] = self.rows[row_ind-1] + len(x_comp)
        self.csr_data = sps.csr_matrix((self.data, self.cols, self.rows), shape=(self.n_rows, self.n_cols))
        self.coeff = self.csr_data[:,0].toarray().reshape(-1)
        self.csr_data = self.csr_data[:,1:]#.toarray()
        self.pick_matrix = sps.csr_matrix((np.ones(self.data.shape[0]), self.cols, self.rows), shape=(self.n_rows, self.n_cols))
        self.pick_matrix = self.pick_matrix[:,1:]
        self.col_nnz = np.array(self.pick_matrix.sum(axis=0)).reshape(-1)
        self.n_cols = self.csr_data.shape[1]


def convert_PSL_to_LP(data):
    A = sps.hstack([data.csr_data, -sps.eye(data.n_rows)]).tocoo()
    c = sps.coo_matrix(np.append(np.zeros(data.n_cols), data.weights)).T
    b = sps.coo_matrix(data.coeff).T
    return A, b, c


def get_lifted_data(LA, Lc, Lb, Bcc, n_orig_vars):
    #BUGBUG: this heavily depends on the fact that the first n columns correspond to lifted RVs. But have a condition to make sure nothing bad happens.
    num_vars = LA.shape[1]-LA.shape[0]
    num_rules = LA.shape[0]
    w_order = LA.tocsr()[:,num_vars:].nonzero()[1]
    new_weights = np.array(Lc[num_vars:]).reshape(-1)[w_order]
    new_csr_data = LA.tocsr()[:,:num_vars]
    new_coeff = np.array(Lb).reshape(-1)
    new_l_fn_pow = train_data.l_fn_pow[:num_rules]
    if new_weights.shape[0] != new_csr_data.shape[0]:
        print ("ERROR. Number of non zero coeff don't match number of rules: {} vs {}".format(new_weights.shape[0], new_csr_data.shape[0]))
        return None
    if Bcc[:n_orig_vars, :(LA.shape[1]-LA.shape[0])].sum() != n_orig_vars:
        print('ERROR: Compression incorrect. Please check the conversion process.')
        return None
    return Data(coeff=new_coeff, csr_data=new_csr_data, l_fn_pow=new_l_fn_pow, weights=new_weights)


train_file = sys.argv[1]#'/Users/sriramsrinivasan/Documents/psl_rerm/Data/libsvm-data/link_prediction_small/trust_pred.txt'
train_data = Data(train_file)
train_data.read()


A, b, c = convert_PSL_to_LP(train_data)


start = time.time()
LA, Lb, Lc, LG, Lh, compresstime, Bcc = saucy.liftAbc(A, b, c, G=A, h=b, sparse=True, orbits=False)
end = time.time()
print('Total time to lift: {}sec, compress time: {}sec'.format(end-start, compresstime));


lifted_data = get_lifted_data(LA, Lc, Lb, Bcc, train_data.n_cols)


print('compression in number of variables - before:{}, after:{}'.format(train_data.n_cols, lifted_data.n_cols))
print('compression in number of rules - before:{}, after:{}'.format(train_data.n_rows, lifted_data.n_rows))


with open(sys.argv[2], 'w') as op:
    for i in range(lifted_data.n_rows):
        s1 = '{} 0:{}'.format(lifted_data.weights[i], lifted_data.coeff[i])
        s2 = ' '.join(['{}:{}'.format(j+1,lifted_data.csr_data[i,j]) for j in lifted_data.pick_matrix[i].nonzero()[1]])
        op.write(s1 + ' ' + s2 + ' # ' + '^1\n')

np.save(file=sys.argv[2]+".converter", arr=Bcc)