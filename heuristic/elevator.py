import time

class Elevator:
    def _init_(self, eid, start_floor=0):
        self.eid = eid
        self.current = start_floor
        self.history = []
        self.time_taken = 0

    def move_to(self, floor):
        distance = abs(floor - self.current)
        self.time_taken += distance
        self.history.append((self.current, floor, distance))
        self.current = floor

    def _str_(self):
        return f"Elevator {self.eid} at floor {self.current}"


def heuristic_choose_elevator(elevators, request):
    # choose elevator with minimum travel distance
    distances = [(abs(e.current - request), e) for e in elevators]
    best = min(distances, key=lambda x: x[0])[1]
    return best


def elevator_system(requests, elevators):
    print("Incoming floor requests:", requests)
    print("Initial elevator positions:")
    for e in elevators:
        print(e)

    while requests:
        req = requests.pop(0)

        # choose best elevator using heuristic
        chosen = heuristic_choose_elevator(elevators, req)
        print(f"\nRequest at floor {req} assigned to {chosen.eid}")

        # move elevator
        chosen.move_to(req)
        print(f"{chosen.eid} moved to floor {req}")

    print("\nFinal elevator positions:")
    for e in elevators:
        print(e)

    print("\nMovement logs:")
    for e in elevators:
        print(f"\nLog for {e.eid}:")
        for s, d, t in e.history:
            print(f"From {s} to {d}, cost = {t} units")
        print("Total time =", e.time_taken)


# Driver code
e1 = Elevator("E1", start_floor=3)
e2 = Elevator("E2", start_floor=8)
e3 = Elevator("E3", start_floor=0)

requests = [2, 10, 5, 7, 1]

elevator_system(requests, [e1, e2, e3])