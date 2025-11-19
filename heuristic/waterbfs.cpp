#include <bits/stdc++.h>
using namespace std;

void bfsWaterJug(int A, int B, int C) {
    queue<pair<int,int>> q;
    set<pair<int,int>> visited;

    q.push({0, 0});
    visited.insert({0, 0});

    while(!q.empty()) {
        auto cur = q.front();
        q.pop();

        int x = cur.first;    // jug1
        int y = cur.second;   // jug2

        cout << "State: (" << x << ", " << y << ")\n";

        // Success
        if (x == C || y == C) {
            cout << "Goal achieved.\n";
            return;
        }

        vector<pair<int,int>> nextStates;

        // Fill operations
        nextStates.push_back({A, y});
        nextStates.push_back({x, B});

        // Empty operations
        nextStates.push_back({0, y});
        nextStates.push_back({x, 0});

        // Pour jug1 -> jug2
        int pour = min(x, B - y);
        nextStates.push_back({x - pour, y + pour});

        // Pour jug2 -> jug1
        pour = min(y, A - x);
        nextStates.push_back({x + pour, y - pour});

        // BFS pushing
        for (auto s : nextStates) {
            if (!visited.count(s)) {
                visited.insert(s);
                q.push(s);
            }
        }
    }

    cout << "No solution found.\n";
}

int main() {
    int A, B, C;
    cout << "Enter Jug1 capacity: ";
    cin >> A;
    cout << "Enter Jug2 capacity: ";
    cin >> B;
    cout << "Enter target amount: ";
    cin >> C;

    bfsWaterJug(A, B, C);
    return 0;
}