import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def triangular(x, a, b, c):
    if a == b and x == a:
        return 1.0
    if b == c and x == c:
        return 1.0
    if x <= a or x >= c:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a) if b != a else 0.0
    if b <= x < c:
        return (c - x) / (c - b) if c != b else 0.0
    return 0.0

def fuzzify_distance(x):
    return {
        'VLD': triangular(x, 0, 0, 100),
        'LD': triangular(x, 50, 150, 300),
        'MD': triangular(x, 200, 500, 1000),
        'HD': triangular(x, 800, 1500, 2000)
    }

def fuzzify_speed(x):
    return {
        'SS': triangular(x, 0, 0, 10),
        'MS': triangular(x, 5, 15, 25),
        'HS': triangular(x, 20, 30, 40),
        'VHS': triangular(x, 35, 50, 50)
    }

def brake_value(label):
    vals = {'low':25.0, 'medium':50.0, 'high':75.0, 'very_high':100.0}
    return vals[label]

fuzzy_rules = {
    ('VLD','SS'):'very_high', ('VLD','MS'):'very_high',
    ('VLD','HS'):'very_high', ('VLD','VHS'):'high',

    ('LD','SS'):'very_high', ('LD','MS'):'high',
    ('LD','HS'):'medium', ('LD','VHS'):'medium',

    ('MD','SS'):'high', ('MD','MS'):'medium',
    ('MD','HS'):'low', ('MD','VHS'):'low',

    ('HD','SS'):'medium', ('HD','MS'):'low',
    ('HD','HS'):'low', ('HD','VHS'):'low'
}

def fuzzy_controller(distance, speed):
    dmf = fuzzify_distance(distance)
    smf = fuzzify_speed(speed)

    weighted = []
    weights = []

    for dk, dv in dmf.items():
        for sk, sv in smf.items():
            strength = min(dv, sv)
            if strength > 0:
                out_label = fuzzy_rules.get((dk, sk), 'low')
                out_val = brake_value(out_label)
                weighted.append(strength * out_val)
                weights.append(strength)

    if len(weights) == 0:
        return 0.0
    
    return sum(weighted) / sum(weights)

def plot_mfs():
    x_dist = np.linspace(0, 2000, 400)
    x_speed = np.linspace(0, 50, 400)

    plt.figure(figsize=(14,5))

    plt.subplot(1,2,1)
    plt.plot(x_dist, [triangular(x,0,0,100) for x in x_dist], label='VLD')
    plt.plot(x_dist, [triangular(x,50,150,300) for x in x_dist], label='LD')
    plt.plot(x_dist, [triangular(x,200,500,1000) for x in x_dist], label='MD')
    plt.plot(x_dist, [triangular(x,800,1500,2000) for x in x_dist], label='HD')
    plt.title('Distance Membership Functions')
    plt.xlabel('Distance (m)')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(x_speed, [triangular(x,0,0,10) for x in x_speed], label='SS')
    plt.plot(x_speed, [triangular(x,5,15,25) for x in x_speed], label='MS')
    plt.plot(x_speed, [triangular(x,20,30,40) for x in x_speed], label='HS')
    plt.plot(x_speed, [triangular(x,35,50,50) for x in x_speed], label='VHS')
    plt.title('Speed Membership Functions')
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_output_mfs(output_value):
    x = np.linspace(0, 100, 300)
    low = [triangular(xx,0,0,40) for xx in x]
    medium = [triangular(xx,25,50,75) for xx in x]
    high = [triangular(xx,50,75,100) for xx in x]
    very_high = [triangular(xx,75,100,100) for xx in x]

    plt.figure(figsize=(8,5))
    plt.plot(x, low, label='Low')
    plt.plot(x, medium, label='Medium')
    plt.plot(x, high, label='High')
    plt.plot(x, very_high, label='Very High')
    plt.axvline(output_value, linestyle='--', color='k', label=f'Output={output_value:.2f}%')
    plt.title('Brake Power Output')
    plt.xlabel('Brake Power (%)')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)
    plt.show()

def surface_plot():
    d_vals = np.linspace(0, 1000, 80)
    s_vals = np.linspace(0, 50, 80)
    D, S = np.meshgrid(d_vals, s_vals)
    Z = np.zeros_like(D)

    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            Z[i,j] = fuzzy_controller(D[i,j], S[i,j])

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(D, S, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_zlabel('Brake Power (%)')
    ax.set_title('Brake Power Surface')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Brake Power (%)')
    plt.show()

if __name__ == "__main__":
    plot_mfs()

    test_cases = [
        (50, 10),
        (50, 40),
        (500, 25),
        (900, 45),
        (200, 20)
    ]

    print("---- Fuzzy Brake Controller Results ----")
    print(f"{'Distance (m)':<15}{'Speed (km/h)':<15}{'Brake Power (%)':<15}")
    print("-" * 45)

    for dist, spd in test_cases:
        out = fuzzy_controller(dist, spd)
        print(f"{dist:<15}{spd:<15}{out:<15.2f}")
        plot_output_mfs(out)

    surface_plot()