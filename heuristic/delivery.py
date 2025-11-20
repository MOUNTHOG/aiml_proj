import math

# Delivery agent class
class DeliveryAgent:
    def __init__(self, aid, start_x=0, start_y=0):
        self.aid = aid
        self.x = start_x
        self.y = start_y
        self.time_taken = 0
        self.history = []

    def distance(self, loc):
        return math.sqrt((self.x - loc[0])**2 + (self.y - loc[1])**2)

    def move_to(self, loc):
        d = self.distance(loc)
        self.time_taken += d
        self.history.append((self.x, self.y, loc[0], loc[1], d))
        self.x, self.y = loc

    def __str__(self):
        return f"{self.aid} at ({self.x}, {self.y})"


# Selects best agent (nearest delivery location)
def choose_best_agent(agents, location):
    distances = [(agent.distance(location), agent) for agent in agents]
    return min(distances, key=lambda x: x[0])[1]


def delivery_system(delivery_points, agents):
    print("Delivery requests:", delivery_points)
    print("\nInitial agent positions:")
    for a in agents:
        print(a)

    for loc in delivery_points:
        best = choose_best_agent(agents, loc)
        print(f"\nLocation {loc} assigned to {best.aid}")
        best.move_to(loc)
        print(f"{best.aid} delivered package at {loc}")

    print("\nFinal agent positions:")
    for a in agents:
        print(a)

    print("\nMovement logs:")
    for a in agents:
        print(f"\nLogs for {a.aid}:")
        for sx, sy, dx, dy, t in a.history:
            print(f"From ({sx},{sy}) to ({dx},{dy}) in time {t:.2f}")
        print("Total time:", round(a.time_taken, 2))


# Driver code
a1 = DeliveryAgent("D1", 0, 0)
a2 = DeliveryAgent("D2", 5, 5)
a3 = DeliveryAgent("D3", 10, 0)

requests = [(2, 1), (6, 4), (9, 2), (1, 3)]

delivery_system(requests, [a1, a2, a3])

