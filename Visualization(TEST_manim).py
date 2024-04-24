from manim import *
import numpy as np

class AtomCollision3D(ThreeDScene):
    def construct(self):
        surface = Rectangle(width=10, height=10, color=BLUE)
        surface = Cutout(surface, fill_opacity=1, color=RED, stroke_color=RED)
        self.add(surface)

        atoms = []
        for i in range(4):
            atom = Sphere(radius=0.2, color=RED)
            atom.move_to(np.random.uniform(-4, 4, 3))
            atoms.append(atom)
            self.add(atom)

        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)

        for atom, row in zip(atoms, y_pred_rf):
            self.play(atom.animate.shift(DOWN * atom.get_z()), run_time=1)
            self.wait(row[1] / 10)

            new_position = atom.get_center() + np.array([row[0], row[1], abs(row[2])])
            self.play(atom.animate.move_to(new_position), run_time=1)

scene = AtomCollision3D()
scene.render(preview=True)