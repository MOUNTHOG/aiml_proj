#include <iostream>
#include <vector>
#include <set>
#include <algorithm> // For min()

using namespace std;

int capA, capB, target;
set<pair<int, int>> visited;
vector<pair<pair<int, int>, string>> path;

bool dfs(int a, int b) {
    // 1. Goal Test (Performance Measure)
    if (a == target || b == target) {
        path.push_back({{a, b}, "Target reached!"});
        return true;
    }

    // 2. Cycle Detection (Prevent infinite loops)
    if (visited.count({a, b})) return false;
    visited.insert({a, b});

    // 3. Try all 6 Actions (Transitions)

    // Action 1: Fill Jug A
    path.push_back({{capA, b}, "Fill Jug A"});
    if (dfs(capA, b)) return true;
    path.pop_back(); // Backtrack

    // Action 2: Fill Jug B
    path.push_back({{a, capB}, "Fill Jug B"});
    if (dfs(a, capB)) return true;
    path.pop_back(); // Backtrack

    // Action 3: Empty Jug A
    path.push_back({{0, b}, "Empty Jug A"});
    if (dfs(0, b)) return true;
    path.pop_back(); // Backtrack

    // Action 4: Empty Jug B
    path.push_back({{a, 0}, "Empty Jug B"});
    if (dfs(a, 0)) return true;
    path.pop_back(); // Backtrack

    // Action 5: Pour A -> B
    {
        int pour = min(a, capB - b);
        path.push_back({{a - pour, b + pour}, "Pour Jug A -> Jug B"});
        if (dfs(a - pour, b + pour)) return true;
        path.pop_back(); // Backtrack
    }

    // Action 6: Pour B -> A
    {
        int pour = min(b, capA - a);
        path.push_back({{a + pour, b - pour}, "Pour Jug B -> Jug A"});
        if (dfs(a + pour, b - pour)) return true;
        path.pop_back(); // Backtrack
    }

    return false;
}

int main() {
    cout << "Enter capacity of Jug A: ";
    cin >> capA;
    cout << "Enter capacity of Jug B: ";
    cin >> capB;
    cout << "Enter target amount: ";
    cin >> target;

    cout << "\nJug A Capacity: " << capA;
    cout << "\nJug B Capacity: " << capB;
    cout << "\nTarget Amount: " << target << "\n\n";

    path.push_back({{0, 0}, "Start"});
    
    // Start the search
    if (dfs(0, 0)) {
        cout << "Solution Path:\n";
        for (auto &step : path) {
            cout << step.second << " -> (" << step.first.first << ", " << step.first.second << ")\n";
        }
    } else {
        cout << "No solution found.\n";
    }
    return 0;
}