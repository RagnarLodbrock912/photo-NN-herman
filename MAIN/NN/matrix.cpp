#include <iostream>
#include <vector>
#include <type_traits>
#include <random>

using namespace std;

long double random(int min, int max) {

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> distrib(min, max);

    return (long double)distrib(gen);
}

double sigmoid(double x, bool is_derivative = false) {

    double sig = 1.0 / (1.0 + exp(-x)); 
    
    if (is_derivative) {
        return sig * (1 - sig);
    }
    
    return sig;
}

double relu(double x, bool is_derivative = false) {
    return (is_derivative) ? ((x > 0) ? 1 : 0) : ((x > 0) ? x : 0);
}

class Matrix
{
private:
    vector<long double> matrix;
    size_t cols;
    size_t rows;
public:
    Matrix(size_t m_cols, size_t m_rows, long double el = 0.f){
        this->cols = m_cols;
        this->rows = m_rows;
        vector<long double> m_matrix(cols * rows, el);
        this->matrix = m_matrix; 
    }

    friend ostream& operator<<(ostream& out, const Matrix& matr) {
        for (int i = 0; i < matr.rows; i++) {
            for (int j = 0; j < matr.cols; j++) {
                out << matr.matrix[i + j] << " ";
            }
            out << endl;
        }
        
        return out;
    }

    
    long double& operator()(size_t row, size_t col) {
        return this->matrix[row * this->cols + col];
    }

    const long double operator()(size_t row, size_t col) const {
        return this->matrix[row * this->cols + col];
    }
    
    Matrix T() {
        Matrix result(this->rows, this->cols);

        for(size_t i = 0; i < this->rows; i++) { 
            for(size_t j = 0; j < this->cols; j++) {
                result(j, i) = this->operator()(i, j);
            }
        }
        
        return result;
    }

    Matrix operator+(const Matrix& other) const {
        if (this->cols != other.cols || this->rows != other.rows) {
            throw out_of_range("Different dimensions");
        }   
        
        Matrix result(this->cols, this->rows);
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                result(i, j) = this->operator()(i, j) + other.matrix[i * cols + j];
            }
        }

        return result;
    }

    Matrix& operator+=(const Matrix& other) {
        if (this->cols != other.cols || this->rows != other.rows) {
            throw out_of_range("Different dimensions");
        }   
        
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                this->operator()(i, j) = this->operator()(i, j) + other.matrix[i * cols + j];
            }
        }

        return *this;
    }
    
    Matrix operator-(const Matrix& other) const {
        if (this->cols != other.cols || this->rows != other.rows) {
            throw out_of_range("Different dimensions");
        }   
        
        Matrix result(this->cols, this->rows);
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                result(i, j) = this->operator()(i, j) - other.matrix[i * cols + j];
            }
        }

        return result;
    }

    Matrix& operator-=(const Matrix& other) {
        if (this->cols != other.cols || this->rows != other.rows) {
            throw out_of_range("Different dimensions");
        }   
        
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                this->operator()(i, j) = this->operator()(i, j) - other.matrix[i * this->cols + j];
            }
        }

        return *this;
    }

    Matrix operator^(const Matrix& other) const { //Harmard product
        if (this->cols != other.cols || this->rows != other.rows) {
            throw out_of_range("Different dimensions");
        }   
        
        Matrix result(this->cols, this->rows);

        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                result(i, j) = this->operator()(i, j) * other.matrix[i * this->cols + j];
            }
        }

        return result;
    }

    Matrix operator*(const long double scalar) const {
        Matrix result(this->cols, this->rows);

        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                result(i, j) = this->operator()(i, j) * scalar;
            }
        }

        return result;
    }

    Matrix operator/=(const long double scalar) const {
        Matrix result(this->cols, this->rows);
        if (scalar == 0) {
            throw invalid_argument("Division by zero");
        }
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                result(i, j) = this->operator()(i, j) / scalar;
            }
        }

        return *this;
    }
    
    Matrix operator*(Matrix& other) const {
        if (this->cols != other.rows) {
            throw invalid_argument("Amount of cols must equal amount of rows for multiplying matrixes");
        }   

        Matrix result(this->cols, other.rows);

        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                result(i, j) = 0;

                for (int k = 0; k < this->cols; k++) {
                    result(i, j) += (this->matrix[i] * other(k, j));
                }
            }
        }

        return result;
    }

    void randomize(double min=-1.0, double max=1.0) {
        for (size_t i = 0; i < this->rows; i++) {
            for (size_t j = 0; j < this->cols; j++) {
                this->operator()(i, j) = ::random(min, max);
            }
        }
    }

    Matrix sigmoid(bool is_derivative = false) const {

        Matrix result(this->cols, this->rows);

        for (size_t i = 0; i < this->rows; i++) {
            for(size_t j = 0; j < this->cols; j++) {
                result(i, j) = ::sigmoid(this->operator()(i, j), is_derivative);
            }
        }

        return result;
    }

    Matrix ReLU(bool is_derivative = false) const {

        Matrix result(this->cols, this->rows);

        for (size_t i = 0; i < this->rows; i++) {
            for(size_t j = 0; j < this->cols; j++) {
                result(i, j) = ::relu(this->operator()(i, j), is_derivative);
            }
        }

        return result;
    }

    long double cross_entropy(const Matrix& ideal) const {

        if (this->rows != ideal.rows || this->cols != ideal.cols) {
            throw invalid_argument("Matrices must be of the same dimension (cross_entropy)\n");
        }
        
        long double res = 0.0;
        for (size_t i = 0; i < this->rows; i++) {
            for (size_t j = 0; j < this->cols; j++) {
                res += -1 *  ideal(i, j) * log(this->operator()(i, j));
            }
        }

        return res;
    }

    Matrix softmax() const {

        Matrix result(this->cols, this->rows);
        long double exp_sum = 0.0f;

        for (size_t i = 0; i < this->rows; i++) {
            for(size_t j = 0; j < this->cols; j++) {
                result(i, j) = exp(this->operator()(i, j));
                exp_sum += result(i, j);
            }
        }

        for (size_t i = 0; i < this->rows; i++) {
            for(size_t j = 0; j < this->cols; j++) {
                result(i, j) /= exp_sum;
            }
        }

        return result;
    }
};

int main() {
    Matrix a(2, 3, 1.0);
    Matrix b(2, 3, 2.0);
    Matrix c(3, 2, 3.0);

    cout << "a + b:\n" << (a + b) << endl;
    cout << "b - a:\n" << (b - a) << endl;
    cout << "a ^ b:\n" << (a ^ b) << endl;
    cout << "a * 5:\n" << (a * 5.0) << endl;
    cout << "Transpose a:\n" << (a.T()) << endl;
    cout << "Sigmoid(a):\n" << (a.sigmoid()) << endl;
    cout << "Sigmoid derivative(a):\n" << (a.sigmoid(true)) << endl;
    cout << "ReLU(a):\n" << (a.ReLU()) << endl;
    cout << "ReLU derivative(a):\n" << (a.ReLU(true)) << endl;
    cout << "Softmax(a):\n" << (a.softmax()) << endl;
    cout << "Cross Entropy(a, ideal=1):\n" << a.cross_entropy(Matrix(2, 3, 1.0)) << endl;

    a.randomize(-1.0, 1.0);
    cout << "Randomized a:\n" << a << endl;

    Matrix d(2, 3, 3.0);
    a += d;
    cout << "a += d:\n" << a << endl;
    a -= d;
    cout << "a -= d:\n" << a << endl;

    cout << "a * c:\n" << (a * c) << endl;

    return 0;
}