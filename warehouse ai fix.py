import numpy as np
import matplotlib.pyplot as plt
import heapq

# Grid setup
grid_size = 10
grid = np.random.uniform(1, 3, size=(grid_size, grid_size))

agent_pos = [0, 0]
battery = 100
steps = 0

# Define package positions: choose 3 lowest value cells (not agent's starting position)
flattened = grid.flatten()
sorted_indices = np.argsort(flattened)
package_positions = [divmod(idx, grid_size) for idx in sorted_indices[1:4]]  # skip [0,0] if it's the lowest

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, goal):
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, (0, start))
    g_scores = {start: 0}
    came_from = {}
    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        closed_list.add(current)
        for d in DIRECTIONS:
            neighbor = (current[0] + d[0], current[1] + d[1])
            if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size and neighbor not in closed_list:
                if grid[neighbor[0], neighbor[1]] > 2.5:  # Avoid obstacles
                    continue
                tentative_g_score = g_scores[current] + 1
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score, neighbor))
                    came_from[neighbor] = current
    return []

def draw_grid(trail=[]):
    plt.figure(figsize=(7,7))
    plt.imshow(grid, cmap='winter', interpolation='nearest')
    plt.colorbar(label="Grid State")
    plt.grid(True)
    # Draw trail
    if trail:
        ys, xs = zip(*trail)
        plt.plot(xs, ys, color='orange', linewidth=2, marker='o', markersize=8, alpha=0.5, label='Trail')
    # Draw agent
    plt.scatter(agent_pos[1], agent_pos[0], color='red', s=150, marker='o', label='Agent')
    # Draw packages
    for y, x in package_positions:
        plt.scatter(x, y, color='black', s=100, marker='s', label='Package' if (y,x)==package_positions[0] else "")
    plt.legend(loc='upper right')
    plt.pause(0.5)
    plt.clf()

# Main loop
trail = [tuple(agent_pos)]
current_package = 0

plt.ion()
while battery > 0 and current_package < len(package_positions):
    goal = package_positions[current_package]
    path = a_star(tuple(agent_pos), goal)
    if not path:  # Can't reach
        print(f"Can't reach package at {goal}")
        current_package += 1
        continue
    for next_pos in path:
        agent_pos = list(next_pos)
        steps += 1
        battery -= 1
        trail.append(tuple(agent_pos))
        draw_grid(trail)
        if battery <= 0:
            break
    current_package += 1

plt.ioff()
draw_grid(trail)
plt.show()

print(f"Total steps taken: {steps}")
print(f"Remaining battery: {battery}")
