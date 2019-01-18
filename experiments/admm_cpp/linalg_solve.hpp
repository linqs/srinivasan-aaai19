#ifndef _LINEAR_ALG_SOLVE_
#define _LINEAR_ALG_SOLVE_
#include <vector>

using namespace std;

void solve(vector<vector<double> > &mat, int Nx, vector<double> &output){
 
    // Triangularization
    for (int i = 0; i < Nx - 1; i++)
        for (int h = i + 1; h < Nx; h++)
        {
            double t = mat[h][i] / mat[i][i];
            for (int j = 0; j <= Nx; j++)
            {
                mat[h][j] = mat[h][j] - t * mat[i][j];
            }
        }

    // Resolution
    for (int i = Nx - 1; i >= 0; i--)
    {
        output[i] = mat[i][Nx];
        for (int j = Nx - 1; j > i; j--)
        {
            output[i] = output[i] - mat[i][j] * output[j];
        }
        output[i] = output[i] / mat[i][i];
    }

}

#endif
