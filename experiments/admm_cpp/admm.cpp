#include <stdlib.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <ctime>
#include <numeric>
#include <functional>
#include <algorithm>
#include <cmath>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include "linalg_solve.hpp"

using namespace std;

typedef struct {
    double primal_r, dual_r, primal_r_th, dual_r_th;
} Residual;

class Helper{
    public:
    static void vec_pow(vector<double> &val, vector<int> &to_pow){
        for (int i = 0; i < val.size(); i++) {
            val[i] = pow(val[i], to_pow[i]);
        }
    }

    static double dot(vector<double> &vec1, vector<double> &vec2){
        if (vec1.size() != vec2.size()){
            cout<<"ERROR: size mismatch in dot product computation."<<vec1.size()<<","<<vec2.size()<<endl;
            exit(1);
        }
        double res = 0;
        for (long i = 0, si=vec1.size(); i < si; i++) {
            res += vec1[i]*vec2[i];
        }
        return res;

    }

    static vector<double> dot(vector<vector<double> > &mat1, vector<double> &vec1, vector<vector<long> > &col_ind){
        vector<double> result(mat1.size(), 0);
        for (long i = 0, si=mat1.size(); i < si; i++) {
            double res = 0;
            if (mat1[i].size() != col_ind[i].size()){
                cout<<"ERROR: size mismatch in dot product computation."<<mat1[i].size()<<","<<col_ind[i].size()<<", in row : "<<i<<endl;
                exit(1); 
            }
            for (long j = 0,lsi=mat1[i].size(); j < lsi; j++) {
                res += mat1[i][j]*vec1[col_ind[i][j]];
            }
            result[i] = res;
        }
        return result;
    }

    template<class T1, class T2>
    static vector<double> diff(vector<T1> &vec1, vector<T2> &vec2) {
        if (vec1.size() != vec2.size()){
            cout<<"ERROR: size mismatch in subtraction computation."<<vec1.size()<<","<<vec2.size()<<endl;
            exit(1);
        }
        vector<double> diff(vec1.size());
        for (long i = 0, si=vec1.size(); i < si ; i++){
            diff[i] = vec1[i] - vec2[i];
        }
        return diff;
    }

    static void truncate_val_to_zero(vector<double> &vec){
        for (long i = 0, si=vec.size(); i < si; i++) {
            vec[i] = (vec[i]<0)?0:vec[i];
        }
    }
    
    static void truncate_val_to_zero_one(vector<double> &vec){
        for (long i = 0, si=vec.size(); i < si; i++) {
            if (vec[i] < 0){
                vec[i] = 0;
            } else if (vec[i] > 1) {
                vec[i] = 1;
            }
            vec[i] = (vec[i]>1)?1:vec[i];
        }
    }

    template<class T>
    static vector<double> divide(vector<T> &vec, double divisor){
        vector<double> res(vec.size());
        for (long i = 0, si=vec.size(); i < si; i++) {
            res[i] = vec[i]/divisor;
        }
        return res;
    }

    template<class T>
    static double norm(vector<T> &vec){
        double v1 = 0;
        for (long i = 0, si=vec.size(); i < si; i++) {
            v1 += vec[i] * vec[i];
        }
        v1 = sqrt(v1);
        return v1;
    }

    template<class T>
    static double norm(vector<vector<T> > &mat){
        double v1 = 0;
        for (long i = 0,si=mat.size(); i < si; i++) {
            for (long j = 0, lsi=mat[i].size(); j < lsi; j++) {
                v1 += mat[i][j] * mat[i][j];
            }
        }
        v1 = sqrt(v1);
        return v1;
    }

};

class Data{
    public:
    vector<long> col_nnz_;
    vector<double> weights_;
    vector<int> power_;
    vector<double> coeff_;
    vector<vector<double> > data_value_;
    vector<vector<long> > data_col_ids_;
    long n_cols_;
    long n_rows_;
    long nnz_;
    const char* data_file_name_;

    void read(){
        cout<<"Reading data file: "<<this->data_file_name_<<endl;
        ifstream data_file(this->data_file_name_);
        if (data_file.is_open()){
            boost::char_separator<char> sep(" #:");
            string line;
            long row_num = 0;
            clock_t start = clock();
            while(getline(data_file,line)) {
                //cout<<line<<endl;
                boost::tokenizer<boost::char_separator<char> > tokens(line, sep);
                size_t count = 0;
                long col_id;
                vector<double> row_vals(0);
                vector<long> row_col_ids(0);
                row_col_ids.reserve(5);
                row_vals.reserve(5);
                long i = 0;
                for (const string& t:tokens) {
                    //cout<<t<<","<<count<<endl;
                    if (count == 0){
                        this->weights_.push_back(stod(t));
                    } else if(boost::starts_with(t, "^")) {
                        string loc_t = t;
                        loc_t.erase(0,1);
                        this->power_.push_back(stoi(loc_t));
                    } else if (count % 2 == 1 ) {
                        col_id = stol(t)-1;
                        if (col_id + 1 > this->n_cols_){
                            this->n_cols_ = col_id + 1;
                            this->col_nnz_.resize(this->n_cols_, 0);
                        }
                    } else {
                        double val = stod(t);
                        if (col_id == -1) {
                            this->coeff_.push_back(val);
                        } else {
                            row_vals.push_back(val);
                            row_col_ids.push_back(col_id);
                            this->col_nnz_[col_id]++;
                            this->nnz_++;
                            i++;
                        }
                    }
                    count++;
                }
                this->data_value_.push_back(row_vals);
                this->data_col_ids_.push_back(row_col_ids);
                row_num++;
            }
            this->n_rows_ = row_num;
            clock_t end = clock();
            data_file.close();
            cout<<"Finished reading. Time taken to read : "<<(end - start) / (double)(CLOCKS_PER_SEC)<<"s"<<endl;
        } else {
            //cout << "Unable to open file";
        }
    }

    Data(const char* data_file_name):col_nnz_(0),
        weights_(0),
        power_(0),
        coeff_(0),
        data_value_(0),
        data_col_ids_(0),
        n_cols_(-1),
        n_rows_(-1),
        nnz_(0){
        this->data_file_name_ = data_file_name;
        read();
    }
    ~Data(){
        for (long i = 0; i < n_rows_; i++) {
            data_value_[i].clear();
            data_col_ids_[i].clear();
        }
        data_value_.clear();
        data_col_ids_.clear();
        weights_.clear();
        power_.clear();
        coeff_.clear();
    }
};

class Admm{
    public:
    vector<vector<double> > rvl_;
    vector<vector<double> > alpha_;
    vector<double> rvg_;
    vector<double> rvg_prev_;
    vector<vector<double> > unit_norm_;
    double lr_;
    int n_ite_;
    double e_abs_;
    double e_rel_;
    Data *data_;

    Admm(Data *data, double lr, int n_ite):
        e_abs_(0.00001), 
        e_rel_(0.001){
        this->data_ = data;
        this->lr_ = lr;
        this->n_ite_ = n_ite;
        this->rvg_prev_.assign(data->n_cols_, 0.0);
        this->rvg_.assign(data->n_cols_, 0.0);
        this->rvl_.reserve(data->n_rows_);
        this->alpha_.reserve(data->n_rows_);
        for (long i = 0; i < data->n_rows_; i++) {
            vector<double> row_rvl(data->data_value_[i].size());
            vector<double> row_alpha(data->data_value_[i].size());
            for (long j = 0, si=data->data_value_[i].size(); j < si; j++) {
                row_rvl[j] = 1;
                row_alpha[j] = 0;
            }
            this->rvl_.push_back(row_rvl);
            this->alpha_.push_back(row_alpha);
        }
        for (int row_num = 0; row_num < data->n_rows_; row_num++) {
            double len = Helper::norm(this->data_->data_value_[row_num]);
            this->unit_norm_.push_back(Helper::divide<double>(this->data_->data_value_[row_num], len));
        }

    }

    ~Admm(){
        for (long i = 0; i < rvl_.size(); i++) {
            rvl_[i].clear();
        }
        rvl_.clear();
        rvg_.clear();
        rvg_prev_.clear();
        for (int i = 0; i < unit_norm_.size(); i++) {
            unit_norm_[i].clear();
        }
        unit_norm_.clear();
    }

    void print_progress(int ite_num, double time_taken){
        Residual r = compute_residual();
        double loss = compute_loss();
        cout<<"header:iteration,time,loss,primal_res,primal_res_th,dual_res,dual_res_th"<<endl;
        cout<<"grepthis:"<<ite_num<<","
            <<time_taken<<","
            <<loss<<","
            <<r.primal_r<<","
            <<r.primal_r_th<<","
            <<r.dual_r<<","
            <<r.dual_r_th<<endl;
    }
    void train(){
        print_progress(0, 0);
        double tot_time = 0.0;
        for (int ite_num = 0; ite_num < this->n_ite_; ite_num++) {
            clock_t start = clock();
            for (long row_num = 0; row_num < this->data_->n_rows_; row_num++) {
                long num_nnz = this->rvl_[row_num].size();
                for (long i = 0; i < num_nnz; i++) {
                    long &col_ind = this->data_->data_col_ids_[row_num][i];
                    this->alpha_[row_num][i] = this->alpha_[row_num][i] + this->lr_ * (this->rvl_[row_num][i] - this->rvg_[col_ind]);
                    this->rvl_[row_num][i] = this->rvg_[col_ind] - (1/this->lr_)*this->alpha_[row_num][i];
                }
                double loss = max(Helper::dot(this->data_->data_value_[row_num], this->rvl_[row_num]) - this->data_->coeff_[row_num], 0.0);
                if (loss == 0) {
                    continue;
                }
                if(this->data_->power_[row_num] == 1){
                    for (int i = 0; i < num_nnz; i++) {
                        this->rvl_[row_num][i] -= (this->data_->weights_[row_num]/this->lr_) * this->data_->data_value_[row_num][i];
                        if (Helper::dot(this->data_->data_value_[row_num], this->rvl_[row_num]) < this->data_->coeff_[row_num]){
                            project(row_num);
                        }
                    }
                } else if (this->data_->power_[row_num] == 2){
                    vector<vector<double> > loc_coeff(num_nnz);
                    for (int i = 0; i < num_nnz; i++) {
                        //Last column is b_co
                        long &col_ind = this->data_->data_col_ids_[row_num][i];
                        loc_coeff[i].resize(num_nnz + 1);
                        loc_coeff[i][num_nnz] = (this->lr_ * this->rvg_[col_ind] - this->alpha_[row_num][i]) + 
                            (2*this->data_->weights_[row_num]*this->data_->data_value_[row_num][i] * 
                             this->data_->coeff_[row_num]);
                        for (int j = 0; j < num_nnz; j++) {
                            loc_coeff[i][j] = 2 * this->data_->weights_[row_num] * 
                                this->data_->data_value_[row_num][i] *  
                                this->data_->data_value_[row_num][j];
                        }
                        loc_coeff[i][i] += this->lr_;
                    }
                    //Solve linear equation.
                    solve(loc_coeff, num_nnz, this->rvl_[row_num]);
                } else{
                    cout<<"ERROR: unsupported loss. Make sure all hinges are linear or quadratic."<<endl;
                    exit(1);
                }
            }
            this->rvg_prev_ = this->rvg_;
            fill(this->rvg_.begin(), this->rvg_.end(), 0);
            for (long i = 0, si=this->rvl_.size(); i < si; i++) {
                for (long j = 0, lsi=this->rvl_[i].size(); j < lsi; j++) {
                    long &col_ind = this->data_->data_col_ids_[i][j];
                    this->rvg_[col_ind] += this->rvl_[i][j] + (this->alpha_[i][j]/this->lr_);
                }
            }
            for (long i = 0; i < this->data_->n_cols_; i++) {
                if (this->data_->col_nnz_[i] == 0){
                    this->rvg_[i] = 0;
                    continue;
                }
                this->rvg_[i] /= this->data_->col_nnz_[i];
            }
            Helper::truncate_val_to_zero_one(this->rvg_);
            double time_per_iter = (clock()-start)/((double)CLOCKS_PER_SEC);
            tot_time += time_per_iter;
            print_progress(ite_num+1, tot_time);
            Residual r = compute_residual();
            if (r.primal_r <= r.primal_r_th && r.dual_r <= r.dual_r_th){
                break;
            }
        }    
    }

    void project(long row_num){
        long num_nnz = this->rvl_[row_num].size();
        if (num_nnz == 1){
            this->rvl_[row_num][0] = this->data_->coeff_[row_num]/this->data_->data_value_[row_num][0];
        } else if (num_nnz == 2){
            double coeff0 = this->data_->data_value_[row_num][0];
            double coeff1 = this->data_->data_value_[row_num][1];
            long col_id0 = this->data_->data_col_ids_[row_num][0];
            long col_id1 = this->data_->data_col_ids_[row_num][1];
            double x0 = this->lr_ * this->rvg_[col_id0] - this->alpha_[row_num][0];
            x0 -= this->lr_ * (coeff0/coeff1)*((-1*this->data_->coeff_[row_num]/coeff1) + this->rvg_[col_id1] - (this->alpha_[row_num][1]/this->lr_));
            x0 /= this->lr_ * (1.0 + ((coeff0*coeff0)/(coeff1*coeff1)));
            double x1 = (this->data_->coeff_[row_num] - coeff0*x0)/coeff1;
            this->rvl_[row_num][0] = x0;
            this->rvl_[row_num][1] = x1;
        } else {
            vector<double> point(num_nnz);
            long first_index = min_element(this->data_->data_col_ids_[row_num].begin(),this->data_->data_col_ids_[row_num].end()) - this->data_->data_col_ids_[row_num].begin();
            double multiplier = -1.0f * this->data_->coeff_[row_num] / this->data_->data_value_[row_num][first_index] * this->unit_norm_[row_num][first_index];
            for (int i = 0; i < num_nnz; i++) {
                point[i] = this->rvg_[this->data_->data_col_ids_[row_num][i]] - (this->alpha_[row_num][i]/this->lr_);
                multiplier += point[i] * this->unit_norm_[row_num][i];
            }

            for (int i = 0; i < num_nnz; i++) {
                this->rvl_[row_num][i] = point[i] - multiplier * this->unit_norm_[row_num][i];
            }
        }
    }

    double compute_loss(){
        vector<double> dot = Helper::dot(this->data_->data_value_, this->rvg_, this->data_->data_col_ids_);
        dot = Helper::diff<double, double>(dot, this->data_->coeff_);
        Helper::truncate_val_to_zero(dot);
        Helper::vec_pow(dot, this->data_->power_);
        double loss = Helper::dot(dot, this->data_->weights_);
        return loss/this->data_->n_rows_;
    }

    Residual compute_residual(){
        return {.primal_r=compute_primal_res(),.dual_r=compute_dual_res(),.primal_r_th=compute_primal_res_th(),.dual_r_th=compute_dual_res_th()};
    }

    double compute_primal_res(){
        double residual = 0;
        for (long i = 0; i < this->data_->n_rows_; i++) {
            for (long j = 0, si=this->data_->data_value_[i].size(); j < si ; j++) {
                double r = this->rvl_[i][j] - this->rvg_[this->data_->data_col_ids_[i][j]];
                residual += r*r;
            }
        }
        return sqrt(residual);
    }


    double compute_dual_res(){
        double dual_res = 0;
        for (long i = 0; i < this->data_->n_cols_; i++) {
            double r = (this->rvg_[i] - this->rvg_prev_[i]);
            dual_res += this->data_->col_nnz_[i] * r * r;
        }
        return this->lr_*sqrt(dual_res);
    }

    double compute_primal_res_th(){
        double res_th = this->e_abs_ * sqrt(this->data_->nnz_);
        double v1 = 0;
        double v2 = Helper::norm<double>(this->rvl_);
        for (long i = 0; i < this->data_->n_cols_; i++) {
            v1 += this->data_->col_nnz_[i] * this->rvg_[i] * this->rvg_[i];
        }
        v1 = sqrt(v1);
        return res_th + this->e_rel_*max(v1, v2);
    }

    double compute_dual_res_th(){
        double res_th = this->e_abs_ * sqrt(this->data_->nnz_);
        double v1 = Helper::norm(this->alpha_);
        return res_th + this->e_rel_*v1;
    }

    void write_variables(const char* output_file_name){
        ofstream op_file(output_file_name);
        if (op_file.is_open()){
            for (int i = 0, si=this->rvg_.size(); i < si; i++) {
                op_file<<i<<"\t"<<this->rvg_[i]<<endl;
            }
            op_file.close();
        }
    }
};


//Test functions:
void test_data_read(const char* data_file_name){
    Data data(data_file_name);
    cout<<"Read file. And stats are: "<<data.n_rows_<<", "<<data.n_cols_<<endl;
    for (long i = 0; i < data.n_rows_; i++) {
        string s = to_string(data.weights_[i]) + " 0:" + to_string(data.coeff_[i]) + " ";
        for (long j = 0; j < data.data_value_[i].size(); j++) {
            s += to_string(data.data_col_ids_[i][j] + 1) + ":" + to_string(data.data_value_[i][j]) + " ";
        }
        s += "^" + to_string(data.power_[i]);
        cout<<s<<endl;
    }
    for (long i = 0; i < data.n_cols_; i++) {
        cout<<data.col_nnz_[i]<<",";
    }
    cout<<endl;
}

int main(int argc, const char *argv[]){
    if (!(argc == 3 || argc ==4)){
        cout<<"ERROR: Number of arguements passed: "<<argc-1<<", required: 2"<<endl;
        cout<<"Usage: ./admm train_file output_file learning_rate(optional, default 0.1)"<<endl;
        exit(0);
    }
    const double learning_rate = (argc==4)?stod(argv[3]):0.1;
    const char* data_file_name = argv[1];
    const char* output_file_name = argv[2];
    Data *data = new Data(data_file_name);
    Admm *admm = new Admm(data, learning_rate, 10000);
    admm->train();
    admm->write_variables(output_file_name);
    delete data;
    delete admm;
    return 0;
}

