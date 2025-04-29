#include <iostream>
#include <vector>
#include <type_traits>
#include <random>
#include <iomanip>

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

class Matrix {
    private:
        vector<long double> matrix;
        size_t cols;
        size_t rows;
    public:
        Matrix(size_t m_cols, size_t m_rows, long double el = 0.0L){
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
            
            long double res = 0.0L;
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

        vector<long double> vectorise() const {
            return this->matrix;
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

            for (size_t layer = 0; layer < layers; layer++) {
                for (size_t col = 0; col < cols; col++) {
                    for (size_t row = 0; row < rows; row++) {
                        flipped(layer, col, row) = this->operator()(layer, this->cols - 1 - col, this->rows - 1 - row);
                    }
                }
            }

            return flipped;
        }

        void randomise(double min=-1.0, double max=1.0) {
            for(size_t i = 0; i < this->tensor.size(); i++) {
                this->tensor[i] = ::random(min, max);
            }
        }

        Tensor sigmoid(bool is_derivative=false) const {
            Tensor result(this->layers, this->cols, this->rows);

            for(size_t i = 0; i < this->tensor.size(); i++) {
                result.tensor[i] = ::sigmoid(this->tensor[i], is_derivative);
            }

            return result;
        }

        Tensor RelU(bool is_derivative=false) const {
            Tensor result(this->layers, this->cols, this->rows);

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

int main() {
    // Тест 1: создание тензора
    cout << "Тест 1: создание тензора 3x3x3" << endl;
    Tensor t(3, 3, 3);
    cout << t << endl;

    // Тест 2: заполним тензор случайными числами и выведем его
    cout << "Тест 2: заполнение случайными числами" << endl;
    t.randomise(-5.0, 5.0);
    cout << t << endl;

    // Тест 3: конволюция
    cout << "Тест 3: конволюция с ядром 2x2" << endl;
    Tensor kernel(6, 2, 2); // ядро 2x2
    kernel.randomise(-1.0, 1.0);
    cout << "Ядро:" << endl;
    cout << kernel << endl;

    Tensor result = t.convolve(kernel);
    cout << "Результат конволюции:" << endl;
    cout << result << endl;

    // Тест 4: поворот тензора на 180 градусов
    cout << "Тест 4: поворот тензора на 180 градусов" << endl;
    Tensor rotated = t.rot180();
    cout << "Тензор после поворота на 180 градусов:" << endl;
    cout << rotated << endl;

    // Тест 5: сигмоида
    cout << "Тест 5: сигмоида" << endl;
    Tensor sigmoidResult = t.sigmoid();
    cout << "Результат применения сигмоиды:" << endl;
    cout << sigmoidResult << endl;

    // Тест 6: ReLU
    cout << "Тест 6: ReLU" << endl;
    Tensor reluResult = t.RelU();
    cout << "Результат применения ReLU:" << endl;
    cout << reluResult << endl;

    // Matrix a(2, 3, 1.0);
    // Matrix b(2, 3, 2.0);
    // Matrix c(3, 2, 3.0);

    // cout << "a + b:\n" << (a + b) << endl;
    // cout << "b - a:\n" << (b - a) << endl;
    // cout << "a ^ b:\n" << (a ^ b) << endl;
    // cout << "a * 5:\n" << (a * 5.0) << endl;
    // cout << "Transpose a:\n" << (a.T()) << endl;
    // cout << "Sigmoid(a):\n" << (a.sigmoid()) << endl;
    // cout << "Sigmoid derivative(a):\n" << (a.sigmoid(true)) << endl;
    // cout << "ReLU(a):\n" << (a.ReLU()) << endl;
    // cout << "ReLU derivative(a):\n" << (a.ReLU(true)) << endl;
    // cout << "Softmax(a):\n" << (a.softmax()) << endl;
    // cout << "Cross Entropy(a, ideal=1):\n" << a.cross_entropy(Matrix(2, 3, 1.0)) << endl;

    // a.randomize(-1.0, 1.0);
    // cout << "Randomized a:\n" << a << endl;

    // Matrix d(2, 3, 3.0);
    // a += d;
    // cout << "a += d:\n" << a << endl;
    // a -= d;
    // cout << "a -= d:\n" << a << endl;

    // cout << "a * c:\n" << (a * c) << endl;

    return 0;
}