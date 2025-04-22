#include <iostream>
#include <vector>

using namespace std;

int main() {
    int width, height;

    cin >> width >> height;
    
    vector<vector<int>> R(height, vector<int>(width, 0));
    vector<vector<int>> G(height, vector<int>(width, 0));
    vector<vector<int>> B(height, vector<int>(width, 0));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cin >> R[i][j];
        }
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cin >> G[i][j];
        }
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cin >> B[i][j];
        }
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cout << R[i][j] << " ";
        }
        cout << endl;
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cout << G[i][j] << " ";
        }
        cout << endl;
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cout << B[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
