// fc.h: numpy arrays from cython , double*
#include <vector>
using namespace std;

vector<double> fc( int N, const double a[], const long int b[], const long int c[], double z[] );
vector<size_t> equitablePartitionSaucyV2(const size_t mvertices, const size_t medges, const double data[], const size_t rown[], const size_t coln[], const size_t b[], int cIters = 0, int coarsest=1);
vector<int> equitablePartitionSaucyBipartite(const size_t nrows, const size_t ncols, const size_t medges, const size_t data[], const size_t rowind[], const size_t colind[], const size_t b[], const size_t c[], int cIters, int coarsest, bool verbose=false);
