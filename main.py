import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Global variables to track collisions
collision_count = 0
collision_events = []
collision_log_file = open("collision_log.txt", "w")

class Particle:
    def __init__(self, position, velocity, radius=1.0, mass=1.0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.radius = radius
        self.mass = mass

def init_particles(N, bounds):
    particles = []
    for _ in range(N):
        position = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=3)
        velocity = np.random.uniform(low=-0.5, high=0.5, size=3)
        mass = np.random.uniform(0.5, 1.5)
        radius = mass ** (1/3)  # Assume constant density
        particles.append(Particle(position, velocity, radius, mass))
    return particles

def compute_forces(particles):
    N = len(particles)
    G = 0.1  # Gravitational constant
    epsilon = 0.1  # Softening parameter to avoid singularity
    forces = [np.zeros(3) for _ in particles]
    for i in range(N):
        for j in range(i+1, N):
            p1 = particles[i]
            p2 = particles[j]
            delta_pos = p2.position - p1.position
            dist = np.linalg.norm(delta_pos)
            dist2 = dist**2 + epsilon**2
            if dist > 1e-2:
                force_mag = G * p1.mass * p2.mass / dist2
                force_dir = delta_pos / dist
                force = force_mag * force_dir
                forces[i] += force
                forces[j] -= force  # Newton's third law
    return forces

def update_positions(particles, dt, bounds):
    forces = compute_forces(particles)
    for p, f in zip(particles, forces):
        acceleration = f / p.mass
        p.velocity += acceleration * dt
        p.position += p.velocity * dt
        # Bounce off the walls
        for i in range(3):
            if p.position[i] < bounds[i, 0]:
                p.position[i] = bounds[i, 0]
                p.velocity[i] *= -1
            if p.position[i] > bounds[i, 1]:
                p.position[i] = bounds[i, 1]
                p.velocity[i] *= -1

def handle_collisions(particles):
    global collision_count, collision_events
    N = len(particles)
    for i in range(N):
        if particles[i] is None:
            continue
        for j in range(i+1, N):
            if particles[j] is None:
                continue
            p1 = particles[i]
            p2 = particles[j]
            delta_pos = p1.position - p2.position
            dist = np.linalg.norm(delta_pos)
            if dist < p1.radius + p2.radius:
                # Merge the two particles
                collision_log_file.write(f"Collision {collision_count}: Stars {i} and {j} merged.\n")
                total_mass = p1.mass + p2.mass
                new_velocity = (p1.mass * p1.velocity + p2.mass * p2.velocity) / total_mass
                new_position = (p1.mass * p1.position + p2.mass * p2.position) / total_mass
                new_radius = (p1.radius**3 + p2.radius**3)**(1/3)
                # Record collision details
                collision_count += 1
                collision_events.append({
                    'collision_number': collision_count,
                    'particles_involved': (i, j),
                    'new_mass': total_mass,
                    'new_radius': new_radius,
                    'position': new_position.copy(),
                    'velocity': new_velocity.copy(),
                })
                # Update particle p1
                p1.mass = total_mass
                p1.velocity = new_velocity
                p1.position = new_position
                p1.radius = new_radius
                particles[j] = None  # Mark p2 for removal

    # Remove merged particles
    particles[:] = [p for p in particles if p is not None]

def animate(i):
    update_positions(particles, dt, bounds)
    handle_collisions(particles)
    ax.clear()
    positions = np.array([p.position for p in particles])
    sizes = np.array([p.radius * 50 for p in particles])  # Scale for visualization
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=sizes)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Star System Simulation')

    # Remove axis panes and grid to declutter the plot
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    # Update text annotations
    collision_count_text.set_text(f"Collisions: {collision_count}    Stars Left: {len(particles)}")

    if collision_events:
        last_collision = collision_events[-1]
        info_text = (
            f"Last Collision:\n"
            f"Particles: {last_collision['particles_involved']}\n"
            f"New Mass: {last_collision['new_mass']:.2f}\n"
            f"New Radius: {last_collision['new_radius']:.2f}"
        )
        last_collision_text.set_text(info_text)
    else:
        last_collision_text.set_text('')

def main():
    global particles, bounds, dt, ax, fig, collision_count_text, last_collision_text

    N = 100
    bounds = np.array([[-50, 50], [-50, 50], [-50, 50]])
    dt = 0.1

    particles = init_particles(N, bounds)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    positions = np.array([p.position for p in particles])
    sizes = np.array([p.radius * 50 for p in particles])  # Scale for visualization
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=sizes)

    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Particle System Simulation')

    # Adjust layout to make room for text
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.85)

    # Initialize text objects
    collision_count_text = fig.suptitle(f"Collisions: {collision_count}    Stars Left: {len(particles)}", fontsize=12)
    last_collision_text = fig.text(0.02, 0.02, '', fontsize=10, ha='left', va='bottom')

    ani = FuncAnimation(fig, animate, frames=1000, interval=30, blit=False)
    plt.show()
    collision_log_file.close()

if __name__ == '__main__':
    main()
