#include <iostream>
#include <vector>
#include <type_traits>
#include <random>
#include <iomanip>
#include <chrono>
#include <functional>
#include <omp.h>

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

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <omp.h>

using namespace std;

long double random(double min = -1.0, double max = 1.0) {
    return min + (max - min) * ((long double)rand() / RAND_MAX);
}

long double sigmoid(long double x, bool derivative = false) {
    if (derivative) {
        long double s = sigmoid(x);
        return s * (1 - s);
    }
    return 1.0L / (1.0L + exp(-x));
}

long double relu(long double x, bool derivative = false) {
    return derivative ? (x > 0 ? 1.0L : 0.0L) : max(0.0L, x);
}

class Matrix {
private:
    vector<long double> matrix;
    size_t cols;
    size_t rows;

public:
    Matrix(size_t m_cols, size_t m_rows, long double el = 0.0L)
        : cols(m_cols), rows(m_rows), matrix(m_cols * m_rows, el) {}

    friend ostream& operator<<(ostream& out, const Matrix& matr) {
        for (size_t i = 0; i < matr.rows; i++) {
            for (size_t j = 0; j < matr.cols; j++) {
                out << matr(i, j) << " ";
            }
            out << endl;
        }
        return out;
    }

    long double& operator()(size_t row, size_t col) {
        return matrix[row * cols + col];
    }

    const long double operator()(size_t row, size_t col) const {
        return matrix[row * cols + col];
    }

    Matrix T() const {
        Matrix result(rows, cols);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(j, i) = (*this)(i, j);
        return result;
    }

    Matrix operator+(const Matrix& other) const {
        if (cols != other.cols || rows != other.rows)
            throw out_of_range("Different dimensions");

        Matrix result(cols, rows);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(i, j) = (*this)(i, j) + other(i, j);
        return result;
    }

    Matrix& operator+=(const Matrix& other) {
        if (cols != other.cols || rows != other.rows)
            throw out_of_range("Different dimensions");

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                (*this)(i, j) += other(i, j);
        return *this;
    }

    Matrix operator-(const Matrix& other) const {
        if (cols != other.cols || rows != other.rows)
            throw out_of_range("Different dimensions");

        Matrix result(cols, rows);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(i, j) = (*this)(i, j) - other(i, j);
        return result;
    }

    Matrix& operator-=(const Matrix& other) {
        if (cols != other.cols || rows != other.rows)
            throw out_of_range("Different dimensions");

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                (*this)(i, j) -= other(i, j);
        return *this;
    }

    Matrix operator^(const Matrix& other) const {
        if (cols != other.cols || rows != other.rows)
            throw out_of_range("Different dimensions");

        Matrix result(cols, rows);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(i, j) = (*this)(i, j) * other(i, j);
        return result;
    }

    Matrix operator*(const long double scalar) const {
        Matrix result(cols, rows);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(i, j) = (*this)(i, j) * scalar;
        return result;
    }

    Matrix operator/=(const long double scalar) const {
        if (scalar == 0)
            throw invalid_argument("Division by zero");

        Matrix result(cols, rows);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(i, j) = (*this)(i, j) / scalar;
        return result;
    }

    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows)
            throw invalid_argument("Invalid dimensions for multiplication");

        Matrix result(other.cols, rows);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < other.cols; j++) {
                long double sum = 0;
                for (size_t k = 0; k < cols; k++)
                    sum += (*this)(i, k) * other(k, j);
                result(i, j) = sum;
            }
        }
        return result;
    }

    void randomize(double min = -1.0, double max = 1.0) {
        #pragma omp parallel
        {
            std::random_device rd;
            std::mt19937 gen(rd() + omp_get_thread_num());
            std::uniform_real_distribution<long double> distrib(min, max);

            #pragma omp for collapse(2)
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    (*this)(i, j) = distrib(gen);
                }
            }
        }
    }

    Matrix sigmoid(bool is_derivative = false) const {
        Matrix result(cols, rows);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(i, j) = ::sigmoid((*this)(i, j), is_derivative);
        return result;
    }

    Matrix ReLU(bool is_derivative = false) const {
        Matrix result(cols, rows);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(i, j) = ::relu((*this)(i, j), is_derivative);
        return result;
    }

    long double cross_entropy(const Matrix& ideal) const {
        if (rows != ideal.rows || cols != ideal.cols)
            throw invalid_argument("Matrix sizes must match");

        long double res = 0.0L;
        #pragma omp parallel for collapse(2) reduction(+:res)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                res += -1.0L * ideal(i, j) * log((*this)(i, j));
        return res;
    }

    Matrix softmax() const {
        Matrix result(cols, rows);
        long double exp_sum = 0.0L;

        #pragma omp parallel for collapse(2) reduction(+:exp_sum)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++) {
                result(i, j) = exp((*this)(i, j));
                exp_sum += result(i, j);
            }

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(i, j) /= exp_sum;
        return result;
    }

    vector<long double> vectorise() const {
        return matrix;
    }
};

class Tensor {
private:
    vector<long double> tensor;
    size_t cols;
    size_t rows;
    size_t layers;
public:
    Tensor(size_t t_layers, size_t t_cols, size_t t_rows, long double el = 0.0) {
        this->tensor = vector<long double>(t_cols * t_rows * t_layers, el);
        this->cols = t_cols;
        this->rows = t_rows;
        this->layers = t_layers;
    }

    long double& operator()(size_t layer, size_t col, size_t row) {
        return tensor[layer * (this->cols  * this->rows) + row * this->cols + col];
    }

    const long double operator()(size_t layer, size_t col, size_t row) const {
        return tensor[layer * (this->cols  * this->rows) + row * this->cols + col];
    }

    Tensor convolve(const Tensor& kernel) const {
        if (kernel.layers % this->layers != 0) {
            throw std::invalid_argument("Kernel layers must be divisible by input tensor layers");
        }

        size_t out_channels = kernel.layers / this->layers;

        Tensor result(
            out_channels,
            this->cols - kernel.cols + 1,
            this->rows - kernel.rows + 1
        );

        #pragma omp parallel for collapse(3)
        for (size_t out_ch = 0; out_ch < out_channels; ++out_ch) {
            for (size_t i = 0; i < result.cols; ++i) {
                for (size_t j = 0; j < result.rows; ++j) {
                    long double sum = 0.0L;

                    for (size_t in_ch = 0; in_ch < this->layers; ++in_ch) {
                        size_t kernel_layer = out_ch * this->layers + in_ch;

                        for (size_t dx = 0; dx < kernel.cols; ++dx) {
                            for (size_t dy = 0; dy < kernel.rows; ++dy) {
                                long double val = this->operator()(in_ch, i + dx, j + dy);
                                long double weight = kernel(kernel_layer, dx, dy);
                                sum += val * weight;
                            }
                        }
                    }

                    result(out_ch, i, j) = sum;
                }
            }
        }

        return result;
    }

    Tensor rot180() {
        Tensor flipped(this->layers, this->cols, this->rows);

        #pragma omp parallel for collapse(3)
        for (size_t layer = 0; layer < layers; layer++) {
            for (size_t col = 0; col < cols; col++) {
                for (size_t row = 0; row < rows; row++) {
                    flipped(layer, col, row) = this->operator()(layer, this->cols - 1 - col, this->rows - 1 - row);
                }
            }
        }

        return flipped;
    }

void randomise(double min = -1.0, double max = 1.0) {
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<long double> distrib(min, max);

        #pragma omp for
        for (size_t i = 0; i < this->tensor.size(); ++i) {
            this->tensor[i] = distrib(gen);
        }
    }
}


    Tensor sigmoid(bool is_derivative=false) const {
        Tensor result(this->layers, this->cols, this->rows);

        #pragma omp parallel for
        for(size_t i = 0; i < this->tensor.size(); i++) {
            result.tensor[i] = ::sigmoid(this->tensor[i], is_derivative);
        }

        return result;
    }

    Tensor RelU(bool is_derivative=false) const {
        Tensor result(this->layers, this->cols, this->rows);

        #pragma omp parallel for
        for(size_t i = 0; i < this->tensor.size(); i++) {
            result.tensor[i] = ::relu(this->tensor[i], is_derivative);
        }

        return result;
    }

    vector<long double> vectorise() const {
        return this->tensor;
    }

    friend ostream& operator<<(ostream& out, const Tensor& t) {
        for (size_t layer = 0; layer < t.layers; ++layer) {
            out << "# Layer " << layer << " #" << endl;
            for (size_t row = 0; row < t.rows; ++row) {
                for (size_t col = 0; col < t.cols; ++col) {
                    out << setw(8) << fixed << setprecision(2)
                        << t(layer, col, row) << " ";
                }
                out << endl;
            }
            out << string(20, '#') << "\n";
        }
        return out;
    }
};

using namespace std::chrono;

void test_time(const string& name, function<void()> func) {
    auto start = high_resolution_clock::now();
    func();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();
    cout << name << " took " << duration << " ms" << endl;
}

int main() {
    const size_t SIZE = 10000;

    Matrix A(SIZE, SIZE);
    Matrix B(SIZE, SIZE);
    A.randomize();
    B.randomize();


    test_time("Matrix randomize", [&]() {
        A.randomize();
        B.randomize();
    });

    test_time("Matrix addition", [&]() {
        Matrix C = A + B;
    });

    test_time("Matrix subtraction", [&]() {
        Matrix C = A - B;
    });

    test_time("Matrix Hadamard (element-wise multiply)", [&]() {
        Matrix C = A ^ B;
    });

    test_time("Matrix scalar multiplication", [&]() {
        Matrix C = A * 3.1415L;
    });

    test_time("Matrix transpose", [&]() {
        Matrix C = A.T();
    });

    test_time("Matrix sigmoid activation", [&]() {
        Matrix C = A.sigmoid();
    });

    test_time("Matrix ReLU activation", [&]() {
        Matrix C = A.ReLU();
    });

    test_time("Matrix softmax", [&]() {
        Matrix C = A.softmax();
    });

    test_time("Cross entropy", [&]() {
        long double loss = A.cross_entropy(B);
    });

    const size_t SMALL_SIZE = 1000;
    Matrix M1(SMALL_SIZE, SMALL_SIZE);
    Matrix M2(SMALL_SIZE, SMALL_SIZE);
    M1.randomize();
    M2.randomize();

    test_time("Matrix multiplication", [&]() {
        Matrix C = M1 * M2.T();
    });

    // srand(time(nullptr));

    // const size_t LAYERS = 3;
    // const size_t SIZE = 1000;

    // Tensor A(LAYERS, SIZE, SIZE);
    // Tensor B(LAYERS, SIZE, SIZE);

    // A.randomise(-1.0, 1.0);
    // B.randomise(-1.0, 1.0);

    // const size_t K_LAYERS = LAYERS * 2;
    // const size_t K_SIZE = 3;
    // Tensor kernel(K_LAYERS, K_SIZE, K_SIZE);
    // kernel.randomise(-1.0, 1.0);

    //     Tensor C = A.convolve(kernel);
    //     (void)C; // чтобы не ругался компилятор
    // });

    // test_time("Tensor rot180", [&]() {
    //     Tensor C = A.rot180();
    //     (void)C;
    // });

    // test_time("Tensor sigmoid activation", [&]() {
    //     Tensor C = A.sigmoid();
    //     (void)C;
    // });

    // test_time("Tensor ReLU activation", [&]() {
    //     Tensor C = A.RelU();
    //     (void)C;
    // });

    // test_time("Tensor randomise", [&]() {
    //     A.randomise(-1.0, 1.0);
    // });

    // test_time("Tensor vectorise and sum", [&]() {
    //     const auto& vec = A.vectorise();
    //     long double sum = 0;
    //     #pragma omp parallel for reduction(+:sum)
    //     for (size_t i = 0; i < vec.size(); i++) {
    //         sum += vec[i];
    //     }
    //     cout << "Sum = " << fixed << setprecision(4) << sum << "\n";
    // });

    return 0;
}